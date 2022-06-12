import logging
import torch
import torch.nn as nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import ACT2FN, BertSelfOutput
import torch.nn.functional as F
from ffn_trainable_module.config import FFNTrainableModuleConfig
logging.basicConfig(level=logging.INFO)

sfile = open('memory_attn.txt', 'w')
n_id = 12 - 1


class MemoryNet(nn.Module):
    def __init__(self, n_memory=50, n_embed=16):
        super().__init__()
        self.memory = nn.Parameter(torch.Tensor(n_memory, n_embed))
        self.k_net = nn.Linear(n_embed, n_embed)
        self.v_net = nn.Linear(n_embed, n_embed)
        self.n_memory = n_memory
        nn.init.normal_(self.memory, 0, 0.02)

    def forward(self, input_embeds):
        batch_size, seq_len, n_embed = input_embeds.size()
        query = input_embeds  # bs, seq_len, n_embed
        key = self.k_net(self.memory)  # n_memory, n_embed
        value = self.v_net(self.memory)

        query_ = query.unsqueeze(-2).repeat([1, 1, self.n_memory, 1])
        # bs, seq_len, n_memory, n_embed
        key_ = key.view([1, 1, self.n_memory, n_embed]).repeat([batch_size, seq_len, 1, 1])
        # bs, seq_len, n_memory, n_embed

        score = F.softmax(torch.sum(query_ * key_, dim=3), dim=2)
        # bs, seq_len, n_memory

        print(score.size())
        print(score.detach().cpu().numpy().tolist(), file=sfile)

        memory = self.memory.view([1, 1, self.n_memory, n_embed]).repeat([batch_size, seq_len, 1, 1])
        # bs, seq_len, n_memory, n_embed
        hidden_states = torch.sum(score.unsqueeze(-1) * self.v_net(memory), dim=2)
        # bs, seq_len, n_embed
        return hidden_states


class BertFFNTrainableModule(nn.Module):
    def __init__(self, config: FFNTrainableModuleConfig):
        super(BertFFNTrainableModule, self).__init__()

        self.down_project = nn.Linear(config.hidden_size, config.ffn_module_size)
        self.up_project = nn.Linear(config.ffn_module_size, config.hidden_size)

        if config.memory_num is not None:
            self.ln_1 = nn.LayerNorm(config.hidden_size,
                                     eps=config.ffn_module_layer_norm_eps)

            self.ln_2 = nn.LayerNorm(config.ffn_module_size,
                                     eps=config.ffn_module_layer_norm_eps)
            self.memory_net = MemoryNet(config.memory_num, config.ffn_module_size)
            self.ln_3 = nn.LayerNorm(config.ffn_module_size,
                                     eps=config.ffn_module_layer_norm_eps)
        else:
            self.activation = ACT2FN[config.ffn_module_act]

        self.memory_num = config.memory_num

        for _, sub_module in self.named_modules():
            self._init_weights(sub_module, config.ffn_module_initializer_range)

    def _init_weights(self, module, initializer_range):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, hidden_states: torch.Tensor, layer_id):
        if self.memory_num is None:
            down_projected = self.down_project(hidden_states)
            activated = self.activation(down_projected)
            up_projected = self.up_project(activated)
            return hidden_states + up_projected
        else:
            hidden_states = self.ln_1(hidden_states)
            down_projected = self.down_project(hidden_states)

            memory_out = self.memory_net(self.ln_2(down_projected))

            memory_out = self.ln_3(memory_out)
            up_projected = self.up_project(memory_out)
            return up_projected


class BertAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output: BertSelfOutput,
                 config: BertFFNTrainableModule,
                 layer_id
                 ):
        super(BertAdaptedSelfOutput, self).__init__()
        self.self_output = self_output
        self.ffn_add_module = BertFFNTrainableModule(config)
        self.layer_id = layer_id

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        residual = hidden_states
        hidden_states = self.ffn_add_module(hidden_states, self.layer_id)
        hidden_states = hidden_states + residual + input_tensor  # bs, seq_len, dim
        hidden_states = self.self_output.LayerNorm(hidden_states)
        return hidden_states


class PrefixNet(nn.Module):
    def __init__(self, n_layer=24, n_head=16,
                 n_embd=1024, mid_dim=1024, preseqlen=50):
        super().__init__()
        self.match_n_layer = n_layer
        self.match_n_head = n_head
        self.match_n_embd = n_embd // n_head
        self.n_embd = n_embd
        self.mid_dim = mid_dim
        self.preseqlen = preseqlen

        self.wte = nn.Embedding(self.preseqlen, n_embd)
        self.input_tokens = torch.arange(self.preseqlen).long()

        self.control = nn.Sequential(
            nn.Linear(n_embd, self.mid_dim),
            nn.Tanh(),
            # nn.Linear(self.mid_dim, self.mid_dim),
            # nn.Tanh(),
            nn.Linear(self.mid_dim, n_layer * 2 * n_embd),
            nn.Dropout(0.1)
        )

    def get_prompt(self, bsz=None, device=None):
        # control_tensor [bsz, n_dim]
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        control_tensor = self.wte(input_tokens)  # [bsz, seqlen, n_dim]
        past_key_values = self.control(control_tensor)
        # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, input_ids=None):
        bsz = input_ids.shape[0]
        device = input_ids.device
        past_key_values = self.get_prompt(bsz=bsz, device=device)
        return past_key_values


def adapt_bert_self_output(config: FFNTrainableModuleConfig, layer_id):
    return lambda self_output: BertAdaptedSelfOutput(self_output, config=config, layer_id=layer_id)


def add_ffn_trainable_part(bert_model: BertModel, config: FFNTrainableModuleConfig, layer_remain_id=0) -> BertModel:
    for layer_id, layer in enumerate(bert_model.encoder.layer):
        if layer_id >= layer_remain_id:
            layer.output = adapt_bert_self_output(config, layer_id)(layer.output)
    return bert_model


def unfreeze_trainable_part(bert_model: nn.Module) -> nn.Module:
    # Unfreeze trainable parts, layer norms and adapter/ffn_memory
    for name, sub_module in bert_model.named_modules():
        if isinstance(sub_module, (BertFFNTrainableModule, nn.LayerNorm, PrefixNet)):
            for param_name, param in sub_module.named_parameters():
                param.requires_grad = True
    return bert_model

