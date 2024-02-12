import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from .transforms import get_transforms
from .dataset import ClassificationDataset
from .utils import compute_class_dist
from .constants import TRAIN_IMG_PATH2019, TRAIN_IMG_PATH2020, TRAIN_LABELS_PATH2019, TRAIN_LABELS_PATH2020, \
                    TRAIN_LABELS_PATH_BALANCED, RANDOM_SEED, IMG_SIZE, _mean, _std


def load_isic_training_data(image_folder2019: str, labels_folder2019: str,
                            image_folder2020: str = None, labels_folder2020: str = None):
    data = pd.read_csv(labels_folder2019)        

    # Category names
    known_category_names = list(data.columns.values[1:3])
    
    # Add path and category columns
    data['path'] = data.apply(lambda row : os.path.join(image_folder2019, row['image_name']+'.jpg'), axis=1)
    data['category'] = np.argmax(np.array(data.iloc[:,1:3]), axis=1)

    if labels_folder2020:
        data2020 = pd.read_csv(labels_folder2020)
        data2020['path'] = data2020.apply(lambda row : os.path.join(image_folder2020, row['image_name']+'.jpg'), axis=1)
        data2020['category'] = np.argmax(np.array(data2020.iloc[:,1:3]), axis=1)

        data = pd.concat([data, data2020]).reset_index()

    return data, known_category_names

def load_ph_test_data(image_folder: str, labels_file: str):
    test_df = pd.read_csv(labels_file)
    # Category names
    known_category_names = list(test_df.columns.values[1:3])
    
    # Add path and category columns
    test_df['path'] = test_df.apply(lambda row : os.path.join(image_folder, row['image_name'] + 
                                                              f"/{row['image_name']}_Dermoscopic_Image/{row['image_name']}" + '.bmp'), axis=1)
    test_df['category'] = np.argmax(np.array(test_df.iloc[:,1:3]), axis=1)

    test_images = test_df['path'].to_list()
    test_targets = test_df['category'].to_numpy()

    _, valid_aug = get_transforms(IMG_SIZE, rgb_mean=_mean, rgb_std=_std)

    test_dataset = ClassificationDataset(image_paths=test_images, 
                                         targets=test_targets,
                                         resize=[IMG_SIZE, IMG_SIZE],
                                         augmentations=valid_aug)

    return test_dataset


def get_train_val(labels: pd.DataFrame, fold: int):
    print('Class weights: ', \
          compute_class_weight(class_weight='balanced', classes=np.unique(labels.nevus), y=labels.nevus))
    
    df_train = labels[labels.kfold != fold]
    df_valid = labels[labels.kfold == fold]
    
    train_images = df_train['path'].to_list()
    train_targets = df_train['category'].to_numpy()
    
    valid_images = df_valid['path'].to_list()
    valid_targets = df_valid['category'].to_numpy()
    
    train_aug, valid_aug = get_transforms(IMG_SIZE, _mean, _std)

    train_dataset = ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        resize=[IMG_SIZE,IMG_SIZE],
        augmentations=train_aug,
    )
    
    valid_dataset = ClassificationDataset(
        image_paths=valid_images,
        targets=valid_targets,
        resize=[IMG_SIZE,IMG_SIZE],
        augmentations=valid_aug,
    )
    
    return train_dataset, valid_dataset


def get_kfold(task: str):
    if task == '2019':
        labels, known_category_names = load_isic_training_data(TRAIN_IMG_PATH2019, 
                                                           TRAIN_LABELS_PATH2019)
    elif task == 'balanced':
        labels, known_category_names = load_isic_training_data(TRAIN_IMG_PATH2019, 
                                                           TRAIN_LABELS_PATH_BALANCED)
    else:
        labels, known_category_names = load_isic_training_data(TRAIN_IMG_PATH2019, TRAIN_LABELS_PATH2019, \
                                                            TRAIN_IMG_PATH2020, TRAIN_LABELS_PATH2020)

    compute_class_dist(labels, known_category_names)
    
    labels["kfold"] = -1
    kf = StratifiedKFold(n_splits=5, 
                         shuffle=True, 
                         random_state=RANDOM_SEED)
    
    for f, (t_, v_) in enumerate(kf.split(X=labels, y=labels["category"])):
        labels.loc[v_, "kfold"] = f

    return labels

    


