import torch
import torch_tensorrt

from src.deploy.constants import TS_MODELS_NUM, TENSORRT_PATH, TENSORRT_FORMAT, TRT_MODELS_NAMES, TRT_MODELS_TYPES
from src.models.base.evaluate import evaluate_ensemble
from src.models.best_model import get_best_model
from src.models.base.model import Model


def get_trt_models() -> tuple[list[Model], list[str]]:
    models, _ = get_best_model()
    
    for i in range(TS_MODELS_NUM):
        model = torch.jit.load(TENSORRT_PATH + TRT_MODELS_NAMES[i] + TENSORRT_FORMAT).cuda()
        models.append(Model(TRT_MODELS_TYPES[i], model))
        
    return models, TRT_MODELS_NAMES[:TS_MODELS_NUM]

 
if __name__ == '__main__':
    get_trt_models()
