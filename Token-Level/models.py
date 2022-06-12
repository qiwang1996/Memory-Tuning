from typing import List

import pytorch_lightning as pl
import torch
from transformers import AutoConfig, AutoTokenizer

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from ffn_trainable_module.bert import PrefixNet

from typing import Union

import time

prefix_attn_file = open('prefix_attn.txt', 'w')
layer = 12 - 1


class BertForTokenClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                if past_key_values is not None:
                    pre_len = past_key_values[0].size(-2)
                    active_loss = attention_mask[:, pre_len:].reshape(-1) == 1
                else:
                    active_loss = attention_mask.reshape(-1) == 1

                active_logits = logits.view(-1, self.num_labels)

                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AutoModelForTokenClassificationFinetuner(pl.LightningModule):
    """
    This class is used to fine-tune a pre-trained model from Transformers using PyTorch-Lightning.
    It is optimized towards models with SentencePiece tokenizers to be converted into LibTorch + SentencePiece
    for C++ deployments.

    :arg model_name: str, pre-trained model name for a model from Transformers
    :arg n_classes: int, number of classes in the classification problem
    :arg max_length: int, maximum length of tokens that the model uses
    :arg lr: float, learning rate for fine-tuning
    :arg eps: float, epsilon parameter for Adam optimizer
    """
    def __init__(self,
                 eval_func,
                 label_enocder,
                 mode: str,
                 model_name: str,
                 n_classes: int,
                 preseqlen: Union[str, None],
                 max_length: int = 60,
                 lr: float = 2e-03,
                 eps: float = 1e-08):
        super(AutoModelForTokenClassificationFinetuner, self).__init__()
        self.max_length = max_length
        self.lr = lr
        self.eps = eps

        config = AutoConfig.from_pretrained(model_name,
                                            num_labels=n_classes,
                                            output_attentions=False,
                                            output_hidden_states=False,
                                            torchscript=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_special_tokens=False)
        self.model = BertForTokenClassification.from_pretrained(model_name, config=config)

        assert mode in ['ft', 'prefix', 'adapter', 'memory-ffn',
                        'memory', 'bitfit', 'mam_adapter']

        if mode != 'mam_adapter' and preseqlen is not None:
            self.prefix_net = PrefixNet(preseqlen=preseqlen)

        self.mode = mode
        self.best_val = -1
        self.eval_func = eval_func
        self.label_encoder = label_enocder

        self.all_time = 0

        self.epochs = 0

    def forward_(self, input_ids, attention_mask=None, labels=None):
        if self.mode in ['ft', 'adapter', 'mam_adapter', 'bitfit', 'memory-ffn']:
            output = self.model(input_ids, attention_mask=attention_mask,
                                labels=labels, return_dict=True)
        elif self.mode == 'prefix' or self.mode == 'memory':
            past = self.prefix_net(input_ids)
            batch_size, _, seq_len, _ = past[0][0].size()
            prefix_mask = torch.ones([batch_size, seq_len]).to(self.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            output = self.model(input_ids, attention_mask=attention_mask,
                                labels=labels, past_key_values=past, return_dict=True)
            '''
            attn = output.attentions
            print(attn[layer][:, :, :, :self.prefix_len].detach().cpu().numpy().tolist(), 
                  file=prefix_attn_file)
            '''
        return output

    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        pre_time = time.time()
        outputs = self.forward_(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
        self.all_time += (time.time() - pre_time)
        self.epochs += 1

        loss, logits = outputs.loss, outputs.logits
        return loss, logits

    def training_step(self, batch, batch_nb):
        loss, logits = self.forward(batch)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        time.sleep(0.3)
        loss, logits = self.forward(batch)

        labels = batch['labels']  #batch_size, seq_len
        _, y_predict = torch.max(logits, dim=2)
        val_metric, _, _ = self.eval_func(y_predict.cpu(),
                                    labels.cpu(), self.label_encoder)

        return {'val_loss': loss, 'val_metric': torch.tensor(val_metric)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_metric = torch.stack([x['val_metric'] for x in outputs]).mean()

        self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('avg_val_metric', avg_val_metric, on_epoch=True, prog_bar=True, sync_dist=True)

        cur_val = torch.mean(self.all_gather(avg_val_metric))
        if self.best_val < cur_val:
            self.best_val = cur_val
            
        print('--------best val metric:{}----------'.format(self.best_val))

    def test_step(self, batch, batch_nb):
        loss, logits = self.forward(batch)

        labels = batch['labels']  #batch_size, seq_len
        _, y_predict = torch.max(logits, dim=2)
        test_metric, _, _ = self.eval_func(y_predict.cpu(),
                                    labels.cpu(), self.label_encoder)

        return {'test_metric': torch.tensor(test_metric)}

    def test_epoch_end(self, outputs):
        avg_test_metric = torch.stack([x['test_metric'] for x in outputs]).mean()
        self.log('avg_test_metric', avg_test_metric, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad],
                                lr=self.lr, eps=self.eps)

