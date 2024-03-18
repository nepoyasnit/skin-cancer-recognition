import onnx
import torch
import torchvision.transforms as T
import tensorrt
import numpy as np
from onnx_helper import ONNXClassifierWrapper,convert_onnx_to_engine
from base.transforms import get_valid_transforms
from PIL import Image
import torch_tensorrt

from base.evaluate import evaluate_ensemble
from base.model import Model
from best_model import get_best_model
from torch2trt import torch2trt


def convert_ensemble():
    models, names = get_best_model()

    PRECISION_TYPE_DIR = 'trt_models/float32/'
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
        torch.jit.save(trt_model, PRECISION_TYPE_DIR + names[i] + '.ts')

        model = torch.jit.load(PRECISION_TYPE_DIR + names[i] + '.ts').cuda()
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
