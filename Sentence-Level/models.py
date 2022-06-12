from typing import List

import pytorch_lightning as pl
import torch
from transformers import AutoConfig, AutoTokenizer, \
    AutoModelForSequenceClassification

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ffn_trainable_module.bert import PrefixNet
from typing import Union

import time


prefix_attn_file = open('prefix_attn.txt', 'w')
layer = 12 - 1


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
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
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
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

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AutoModelForSequenceClassificationFinetuner(pl.LightningModule):
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
                 mode: str,
                 model_name: str,
                 n_classes: int,
                 preseqlen: Union[str, None],
                 max_length: int = 100,
                 lr: float = 2e-03,
                 eps: float = 1e-08):
        super(AutoModelForSequenceClassificationFinetuner, self).__init__()
        self.max_length = max_length
        self.lr = lr
        self.eps = eps

        config = AutoConfig.from_pretrained(model_name,
                                            num_labels=n_classes,
                                            output_attentions=False,
                                            output_hidden_states=False,
                                            torchscript=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = BertForSequenceClassification.from_pretrained(model_name, config=config)

        assert mode in ['ft', 'prefix', 'adapter', 'memory-ffn',
                        'memory', 'bitfit', 'mam_adapter']

        if mode != 'mam_adapter' and preseqlen is not None:
            self.prefix_net = PrefixNet(preseqlen=preseqlen)

        self.mode = mode
        
        self.best_val = -1

        self.eval_func = eval_func

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
            print(attn[layer][:, :, 0, :self.prefix_len].detach().cpu().numpy().tolist(), 
                                                        file=prefix_attn_file)
            '''

        return output

    def tokenize(self, texts: List[str]):
        x_tokenized = self.tokenizer(texts, padding=True, truncation=True,
                                     max_length=self.max_length,
                                     return_tensors='pt')
        input_ids = x_tokenized['input_ids'].to(self.device)
        attention_mask = x_tokenized['attention_mask'].to(self.device)
        return input_ids, attention_mask

    def forward(self, batch):
        x = batch['x']
        y = batch.get('y', None)

        pre_time = time.time()
        outputs = self.forward_(*self.tokenize(x), labels=y)
        self.all_time += (time.time() - pre_time)
        self.epochs += 1

        if y is not None:
            loss, logits = outputs.loss, outputs.logits
        else:
            loss = None
            logits = outputs.logits
        return loss, logits

    def training_step(self, batch, batch_nb):
        loss, logits = self.forward(batch)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        time.sleep(0.3)
        loss, logits = self.forward(batch)

        y = batch['y']
        a, y_hat = torch.max(logits, dim=1)
        val_metric = self.eval_func(y_hat.cpu(), y.cpu())

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

        y = batch['y']
        a, y_hat = torch.max(logits, dim=1)
        test_metric = self.eval_func(y_hat.cpu(), y.cpu())

        return {'test_metric': torch.tensor(test_metric)}

    def test_epoch_end(self, outputs):
        avg_test_metric = torch.stack([x['test_metric'] for x in outputs]).mean()
        self.log('avg_test_metric', avg_test_metric, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad],
                                lr=self.lr, eps=self.eps)

