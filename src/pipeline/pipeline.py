import os
import torch
import numpy as np
from datetime import datetime
from PIL import Image

from src.models.base.transforms import get_valid_transforms
from src.models.best_model import get_best_model
from src.utils.attention_maps import get_attention_map

from src.pipeline.constants import DEFAULT_SAVE_PATH, DEFAULT_IMG_NAME, IMG_FORMAT, IMG_SIZE, CUDA_DEVICE ,\
                                CPU_DEVICE
from src.inspyrenet.run.Inference import get_mask_image


class ImagePipeline:
    def __init__(self, _image: Image, _image_name: str = None):
        self.orig_image = _image
        self.image_name = _image_name
        self.device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)
        self.models, self.models_names = get_best_model()

    def process(self):
        '''
         Get model output and attention maps
        '''
        os.makedirs(DEFAULT_SAVE_PATH, exist_ok=True)

        if self.image_name:
            image_path = DEFAULT_SAVE_PATH + self.image_name
            self.orig_image.save(image_path)
        else:
            image_path = DEFAULT_SAVE_PATH + DEFAULT_IMG_NAME + \
                            datetime.now().strftime("%d_%m_%Y_%H:%M:%S") + IMG_FORMAT
            self.orig_image.save(image_path)

        image = self._transform_image(image_path)  

        output = self._predict(image)

        attention_img = get_attention_map(self.models[-1], self.orig_image)

        return output, attention_img

    def _transform_image(self, image_path: str):
        '''
         Make crops and albumentations transforms 
        '''
        mask = get_mask_image(image_path, self.device)[0]
        image = self.__crop_image(self.orig_image, mask)

        transforms = get_valid_transforms(IMG_SIZE)
        image = transforms(image=np.float32(image))['image'].astype(np.float32)
        image = image.transpose(2, 0, 1)
        
        return image

    def _predict(self, image: Image):
        '''
         Get prediction from models ensemble
        '''
        for model in self.models:
            model.to(self.device)
            model.eval()

        image = torch.from_numpy(image).unsqueeze(0)
        image = image.to(self.device)

        output = torch.zeros([1, 2]).to(self.device)
        with torch.no_grad():
            for model in self.models:
                output += model.forward(image)
            
            output /= len(self.models)

        return output


    def __crop_image(self, image: Image, mask: Image):
        '''
         Crop image by mask
        '''
        bbox = mask.getbbox()

        cropped_image = image.crop(bbox)

        return cropped_image
