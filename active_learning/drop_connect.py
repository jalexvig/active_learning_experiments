import numpy as np
import torch
from sklearn_extra.cluster import KMedoids
from torch import nn, optim

from deepcure.evaluation.metrics import calculate_regression_metrics
from active_learning.utils import convert_df_to_torch


from torch.nn import Parameter

import torch


def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + "_raw", Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + "_raw")
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, "forward", forward)


class WeightDrop(torch.nn.Module):
    def __init__(self, module: nn.Module, weights, dropout: float = 0.0):
        super(WeightDrop, self).__init__()
        _weight_drop(module, weights, dropout)
        self.forward = module.forward
