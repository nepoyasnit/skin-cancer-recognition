import torch
import torch_tensorrt

from src.models.base.evaluate import evaluate_ensemble
from src.models.best_model import get_best_model
from src.models.base.model import Model
from src.deploy.constants import TENSORRT_FORMAT, TENSORRT_PATH, TS_MODELS_NUM, TRT_MODELS_NAMES, \
                        MODEL_INPUT_DIM, TRT_MODELS_TYPES


def __test_models(models: list):
    test_loss, test_w_f1, test_sens, test_spec, test_acc = evaluate_ensemble(models)
    print(f" \
            Test loss: {test_loss}\n \
            Test F1: {test_w_f1}\n \
            Test sensitivity: {test_sens}\n \
            Test specificity: {test_spec}\n \
            Test accuracy: {test_acc}\n \
          ")
    

def _export_models(models):
  x = torch.ones(MODEL_INPUT_DIM).cuda()

  for i in range(TS_MODELS_NUM):
    models[i].eval().cuda()

    trt_model = torch_tensorrt.compile(models[i].model, ir="ts", inputs=[x], enabled_precisions={torch.float16})
    torch.jit.save(trt_model, TENSORRT_PATH + TRT_MODELS_NAMES[i] + TENSORRT_FORMAT)

def _import_models():
  models = []
  for i in range(TS_MODELS_NUM):
      model = torch.jit.load(TENSORRT_PATH + TRT_MODELS_NAMES[i] + TENSORRT_FORMAT).cuda()
      models.append(Model(TRT_MODELS_TYPES[i], model))

  return models


def convert_ensemble():
    models, _ = get_best_model()
    
    _export_models(models)
    models = _import_models()    

 
if __name__ == '__main__':
    convert_ensemble()
