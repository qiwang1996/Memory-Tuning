from typing import NamedTuple, Union, Callable
import torch.nn as nn


class FFNTrainableModuleConfig(NamedTuple):
    hidden_size: int
    ffn_module_size: int
    ffn_module_act: Union[str, Callable]
    ffn_module_initializer_range: float
    ffn_module_layer_norm_eps: float
    memory_num: Union[int, None]


def freeze_all_parameters(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False
    return model
