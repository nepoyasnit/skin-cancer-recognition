import torch

from src.models.base.model import Model
from src.models.constants import CUDA_DEVICE, BEST_MODELS_NAMES, BEST_MODELS_TYPES, BEST_MODELS_DIR, WEIGHTS_FORMAT


def get_best_model():
    device = torch.device(CUDA_DEVICE)
    models = []

    for model_name, model_type in zip(BEST_MODELS_NAMES, BEST_MODELS_TYPES):
        model = Model(model_type, pretrained=True)
        model.load_state_dict(torch.load(BEST_MODELS_DIR + model_name + WEIGHTS_FORMAT, map_location=device))
        models.append(model)    

    return models, BEST_MODELS_NAMES