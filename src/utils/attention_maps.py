from torchvision.models import *
from src.gradcam.visualisation.core.utils import device
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch 
from src.gradcam.utils import *
from PIL import Image
import cv2

from src.gradcam.visualisation.core.utils import image_net_postprocessing

from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
from src.gradcam.visualisation.core import *
from src.gradcam.visualisation.core.utils import image_net_preprocessing
from matplotlib.animation import FuncAnimation
from collections import OrderedDict

from src.models.base.model import Model
from src.models.constants import IMG_SIZE, EFFICIENTVIT_TIMM, BEST_MODELS_DIR, WEIGHTS_FORMAT, CUDA_DEVICE, CPU_DEVICE, ATTENTION_MODEL


def get_attention_map(model:Model, image: Image):
    device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)
    
    inputs  = [Compose([Resize((IMG_SIZE,IMG_SIZE)), ToTensor(), image_net_preprocessing])(image).unsqueeze(0)]  # add 1 dim for batch
    inputs = [i.to(device) for i in inputs]

    model.to(device)
    model.eval()

    visualizer = GradCam(model, device)

    model_out = list(map(lambda x: tensor2img(visualizer(x, None,postprocessing=image_net_postprocessing)[0]), inputs))

    model_out = model_out[0] * 255
    model_out = model_out.astype(np.uint8)
    img = Image.fromarray(model_out)
   
    return img


if __name__ == '__main__':
    image = Image.open('../PH2Dataset/PH2 Dataset images/IMD002/IMD002_Dermoscopic_Image/IMD002.bmp')
    
    model = Model(EFFICIENTVIT_TIMM,pretrained=True)
    model.load_state_dict(torch.load(BEST_MODELS_DIR + ATTENTION_MODEL + WEIGHTS_FORMAT, map_location=device))
    
    get_attention_map(model, image).save('result.jpg')
