import torch


def example_model_util_function(x):
    return torch.cat([x, x, x], dim=-1)