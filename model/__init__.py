import torch

from model import model as global_models


def create_model(cfg):
    model = getattr(global_models, cfg["name"])(cfg)
    model.to(cfg["device"] if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("GPU not available, using CPU instead.")

    return model