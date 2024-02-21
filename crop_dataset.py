import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from PIL import Image

def crop_image_with_mask(image_path, mask_path, output_path):
    # Загрузка исходной картинки
    image = Image.open(image_path)
    
    # Загрузка маски
    mask = Image.open(mask_path).convert("L")
    
    # Получение ограничивающей рамки маски
    bbox = mask.getbbox()
    
    # Обрезка картинки по ограничивающей рамке
    cropped_image = image.crop(bbox)
    
    # Сохранение обрезанной картинки
    cropped_image.save(output_path)

# Пример использования
def crop_isic2019():
    labels = pd.read_csv("../isic2019/labels/official/binary_labels2019_2cls.csv")
    cropped = labels.copy(deep=True)

    labels['path'] = labels.apply(lambda row : os.path.join("../isic2019/images/official/", row['image_name'] + '.jpg'), axis=1)
    labels['category'] = np.argmax(np.array(labels.iloc[:,1:3]), axis=1)

    cropped['path'] = cropped.apply(lambda row : os.path.join("../isic2019_mask/images/official/", row['image_name'] + '.png'), axis=1)
    cropped['category'] = np.argmax(np.array(cropped.iloc[:,1:3]), axis=1)

    for image_name, image_path, mask_path in zip(labels.image_name, labels.path, cropped.path):
        crop_image_with_mask(image_path, mask_path, "../isic2019_crop/images/official/" + image_name + '.jpg')
        

    #crop_image_with_mask(image_path, mask_path, output_path)   



crop_isic2019()

