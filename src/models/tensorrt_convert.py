import torch
import torch_tensorrt

from src.models.base.evaluate import evaluate_ensemble
from src.models.best_model import get_best_model
from src.models.constants import TENSORRT_FORMAT, TENSORRT_PATH


def convert_ensemble():
    models, names = get_best_model()

    PRECISION_TYPE_DIR = TENSORRT_PATH
    BATCH_SIZE = 1
    IMG_SIZE = 224
    x = torch.ones((120, 3, 224, 224)).cuda()

    test_loss, test_w_f1, test_sens, test_spec, test_acc = evaluate_ensemble(models)
    print(f" \
            Test loss: {test_loss}\n \
            Test F1: {test_w_f1}\n \
            Test sensitivity: {test_sens}\n \
            Test specificity: {test_spec}\n \
            Test accuracy: {test_acc}\n \
          ")       
    
    for i in range(3):
        models[i].eval().cuda()

        trt_model = torch_tensorrt.compile(models[i].model, ir="ts", inputs=[x], enabled_precisions={torch.float8_e5m2})
        torch.jit.save(trt_model, PRECISION_TYPE_DIR + names[i] + TENSORRT_FORMAT)

        model = torch.jit.load(PRECISION_TYPE_DIR + names[i] + TENSORRT_FORMAT).cuda()
        models[i].model = model
    
    test_loss, test_w_f1, test_sens, test_spec, test_acc = evaluate_ensemble(models)
    print(f" \
            Test loss: {test_loss}\n \
            Test F1: {test_w_f1}\n \
            Test sensitivity: {test_sens}\n \
            Test specificity: {test_spec}\n \
            Test accuracy: {test_acc}\n \
          ")

 
if __name__ == '__main__':
    convert_ensemble()
