from glob import glob
import pandas as pd

import cv2
from skimage import io
import albumentations as A
import torch
import torchvision.transforms as transforms
import os
import random
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from torch.nn import functional as F
from glob import glob
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch import nn
from torchsummary import summary
import warnings
import time
from datetime import datetime

import gc
import sys

from collections import Counter

from PIL import Image
from PIL import ImageFile

import timm
import timm.loss
import timm.optim
import timm.utils
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from torchmetrics.functional import f1_score
from timm.scheduler import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter


# import torch_xla
# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.xla_multiprocessing as xmp
# import torch_xla.distributed.parallel_loader as pl

# import warnings
# warnings.filterwarnings('ignore')


CORE_PATH = ""
DATA2019_PATH = "../../isic2019/labels/official/binary_labels2019_2cls.csv"
DATA2020_PATH = "../../isic2020/labels/binary_labels2020_2cls.csv"
TRAIN_IMG2019_PATH = "../../isic2019/images/official/"
TRAIN_IMG2020_PATH = "../../isic2020/images/"

RANDOM_SEED = 21
IMG_SIZE = 224
BATCH_SIZE = 20
LR = 2e-05
GAMMA = 0.7
N_EPOCHS = 5

_mean = np.array([0.6237459654304592, 0.5201169854503829, 0.5039494477029685])
_std = np.array([0.24196317678786788, 0.2233599432947672, 0.23118716487089888])



def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()


def load_isic_training_data(image_folder2019=None, image_folder2020=None, labels2019=None, labels2020=None):
    if not labels2019 and not labels2020:    
        raise Exception("Add one or both datasets!")

    if labels2019:
        data2019 = pd.read_csv(labels2019)
    
    if labels2020:
        data2020 = pd.read_csv(labels2020)
        

    # Category names
    known_category_names = list(data2019.columns.values[1:3])
    
    # Add path and category columns
    data2019['path'] = data2019.apply(lambda row : os.path.join(image_folder2019, row['image_name']+'.jpg'), axis=1)
    data2019['category'] = np.argmax(np.array(data2019.iloc[:,1:3]), axis=1)

    data2020['path'] = data2020.apply(lambda row : os.path.join(image_folder2020, row['image_name']+'.jpg'), axis=1)
    data2020['category'] = np.argmax(np.array(data2020.iloc[:,1:3]), axis=1)

    data_csv = pd.concat([data2019, data2020]).reset_index()

    return data_csv, known_category_names

def compute_class_dist(df,known_category_names):
    sample_count_train = df.shape[0]
    count_per_category_train = Counter(df['category'])
    for i, c in enumerate(known_category_names):
        print("'%s':\t%d\t(%.2f%%)" % (c, count_per_category_train[i], count_per_category_train[i]*100/sample_count_train))

    return


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths,
        targets,
        resize,
        augmentations=None,
        backend="pil",
        channel_first=True,
    ):
        """
        :param image_paths: list of paths to images
        :param targets: numpy array
        :param resize: tuple or None
        :param augmentations: albumentations augmentations
        """
        super().__init__()
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.backend = backend

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        targets = self.targets[item]
        if self.backend == "pil":
            image = Image.open(self.image_paths[item]).convert("RGB")
            #             image = np.array(image)
            if self.augmentations is not None:
                image = self.augmentations(image)
        else:
            raise Exception("Backend not implemented")
        return image, targets


def get_transforms(image_size, rgb_mean, rgb_std):

#     transforms_train = A.Compose([
#         A.Resize(image_size, image_size),
#         A.Normalize(rgb_mean, rgb_std, max_pixel_value=255.0, always_apply=True),
#         A.Transpose(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.HorizontalFlip(p=0.5),
#         A.RandomBrightness(limit=0.2, p=0.75),
#         A.RandomContrast(limit=0.2, p=0.75),
#         A.OneOf([
#             A.MotionBlur(blur_limit=5),
#             A.MedianBlur(blur_limit=5),
#             A.GaussianBlur(blur_limit=5),
#             A.GaussNoise(var_limit=(5.0, 30.0)),
#         ], p=0.7),

#         A.OneOf([
#             A.OpticalDistortion(distort_limit=1.0),
#             A.GridDistortion(num_steps=5, distort_limit=1.),
#             A.ElasticTransform(alpha=3),
#         ], p=0.7),

# #         A.CLAHE(clip_limit=4.0, p=0.7),
#         A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
#         A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
#         A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
#         A.pytorch.ToTensorV2()
#     ])

#     transforms_val = A.Compose([
#         A.Resize(image_size, image_size),
#         A.Normalize(rgb_mean, rgb_std, max_pixel_value=255.0, always_apply=True),
#         A.pytorch.ToTensorV2()
#     ])

    # create image augmentations
    transforms_train = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std),
        ]
    )

    transforms_valid = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std),
        ]
    )

    return transforms_train, transforms_valid

def get_train_val(fold):
    df_train = df_ground_truth[df_ground_truth.kfold != fold]
    df_valid = df_ground_truth[df_ground_truth.kfold == fold]
    
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

def get_whole_dataset():

    images = df_ground_truth['path'].to_list()
    targets = df_ground_truth['category'].to_numpy()
    
    _,valid_aug = get_transforms(IMG_SIZE, _mean, _std)

    dataset = ClassificationDataset(
        image_paths=images,
        targets=targets,
        resize=[IMG_SIZE,IMG_SIZE],
        augmentations=valid_aug,
    )

    
    return dataset


class Model(nn.Module):
    def __init__(self, timm_model_name, n_classes=2, pretrained=False):

        super(Model, self).__init__()
        self.num_classes = n_classes

        self.model = timm.create_model(
            timm_model_name,
#             "mobilevitv2_200_384_in22ft1k",
            pretrained=pretrained,
            num_classes=self.num_classes,
        )

        # self.model.head = nn.Linear(self.model.head, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x

    def train_one_epoch(self, train_loader, criterion, optimizer, device):
        # keep track of training loss
        epoch_loss = 0.0
        epoch_w_f1 = 0.0

        ###################
        # train the model #
        ###################
        self.model.train()
        for i, (data, target) in enumerate(train_loader):
            # move tensors to GPU if CUDA is available
            if device.type == "cuda":
                data, target = data.cuda(), target.cuda()
            elif device.type == "xla":
                data = data.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.int64)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.forward(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Calculate Weighted F1
            w_f1 = f1_score(
                output, target, num_classes=self.num_classes, average="weighted", task="multiclass"
            )
            output = output.to('cpu').detach().numpy()
            target = target.to('cpu').detach().numpy()

            tn, fp, fn, tp = confusion_matrix(target, np.argmax(output, 1), labels=[0,1]).ravel()
            # sensitivity = tp/(tp+fn)
            # specificity = tn/(tn+fp)
            acc_computed = (tp+tn)/(tn+fp+fn+tp)
            # torchmetrics.functional.f1(output,target,num_classes=len(known_category_names),average='weighted')
            # update training loss and accuracy
            epoch_loss += loss
            epoch_w_f1 += w_f1

            # perform a single optimization step (parameter update)
            if device.type == "xla":
                xm.optimizer_step(optimizer)
                if i % 20 == 0:
                    xm.master_print(f"\tBATCH {i+1}/{len(train_loader)} - LOSS: {loss}")

            else:
                optimizer.step()
                if i % 20 == 0:
                    print(f"\tBATCH {i+1}/{len(train_loader)} - LOSS: {loss}")
                    print("Accuracy: ", acc_computed)
                    
        epoch_loss.to('cpu').detach().numpy()
        epoch_w_f1.to('cpu').detach().numpy()

        return epoch_loss / len(train_loader), epoch_w_f1 / len(train_loader)

    def validate_one_epoch(self, valid_loader, criterion, device, beta=0.000000001):
        # keep track of validation loss
        valid_loss = 0.0
        valid_w_f1 = 0.0
        sensitivity = 0.0
        specificity = 0.0
        acc_computed = 0.0
        nan_vals = 0

        ######################
        # validate the model #
        ######################
        self.model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if device.type == "cuda":
                data, target = data.cuda(), target.cuda()
            elif device.type == "xla":
                data = data.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.int64)

            with torch.no_grad():
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)

                # calculate the batch loss
                loss = criterion(output, target)
                # Calculate Weighted F1
                w_f1 = f1_score(
                    output, target, num_classes=self.num_classes, average="weighted", task="multiclass"
                )
                output = output.to('cpu').detach().numpy()
                target = target.to('cpu').detach().numpy()

                #print(confusion_matrix(target, np.argmax(output, 1), labels=[0,1]).ravel())

                tn, fp, fn, tp = confusion_matrix(target, np.argmax(output, 1), labels=[0,1]).ravel()
                if tp + fn == 0:
                    nan_vals += 1

                sensitivity += tp/(tp+fn + beta)
                specificity += tn/(tn+fp + beta)
                acc_computed += (tp+tn)/(tn+fp+fn+tp)

                # update average validation loss and accuracy
                valid_loss += loss
                valid_w_f1 += w_f1
        
        valid_loss = valid_loss.cpu().numpy()
        valid_w_f1 = valid_w_f1.cpu().numpy()
        print(nan_vals, len(valid_loader))
        
        return valid_loss / len(valid_loader), valid_w_f1 / len(valid_loader), sensitivity / len(valid_loader), \
                specificity / len(valid_loader), acc_computed / len(valid_loader)


def fit_gpu(
    model, epochs, device, criterion, optimizer, train_loader, valid_loader=None
):

    valid_loss_min = np.Inf  # track change in validation loss

    # keeping track of losses as it happen
    train_losses = []
    valid_losses = []
    valid_sensitivity = []
    valid_specificity = []
    valid_accuracy = []
    train_f1s = []
    valid_f1s = []

    for epoch in range(1, epochs + 1):
        gc.collect()
        #         para_train_loader = pl.ParallelLoader(train_loader, [device])

        print(f"{'='*50}")
        print(f"EPOCH {epoch} - TRAINING...")
        train_loss, train_w_f1 = model.train_one_epoch(
            train_loader, criterion, optimizer, device
        )
        print(
            f"\n\t[TRAIN] EPOCH {epoch} - LOSS: {train_loss}, WEIGHTED F1: {train_w_f1}\n"
        )
        train_losses.append(train_loss)
        train_f1s.append(train_w_f1)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("F1/train", train_w_f1, epoch)

        gc.collect()

        if valid_loader is not None:
            gc.collect()
            #         para_valid_loader = pl.ParallelLoader(valid_loader, [device])
            print(f"EPOCH {epoch} - VALIDATING...")
            valid_loss, valid_w_f1, sensitivity, specificity, accuracy = model.validate_one_epoch(
                valid_loader, criterion, device
            )
            print(type(valid_loss), type(valid_w_f1))
            print(f"\t[VALID] LOSS: {valid_loss},  WEIGHTED F1: {valid_w_f1}\n")
            print('Sensitivity: ', sensitivity)
            print('Specificity: ', specificity)
            print('Accuracy: ', accuracy)
            valid_losses.append(valid_loss)
            valid_f1s.append(valid_w_f1)
            valid_sensitivity.append(sensitivity)
            valid_specificity.append(specificity)
            valid_accuracy.append(accuracy)

            writer.add_scalar("Loss/val", valid_loss, epoch)
            writer.add_scalar("F1/val", valid_w_f1, epoch)
            writer.add_scalar("Specificity/val", specificity, epoch)
            writer.add_scalar("Sensitivity/val", sensitivity, epoch)
            writer.add_scalar("Accuracy/val", accuracy, epoch)

            gc.collect()

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min and epoch != 1:
                print(
                    "Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...".format(
                        valid_loss_min, valid_loss
                    )
                )
            torch.save(
                model.state_dict(),
                f'weights/checkpoints/efficientvit-merged/efficientvit_m5_{epoch}_{datetime.now().strftime("%Y%m%d-%H%M")}.pth',
            )
            valid_loss_min = valid_loss
    
    return {
        "train_loss": train_losses,
        "valid_losses": valid_losses,
        "train_w_f1": train_f1s,
        "valid_w_f1": valid_f1s,
        "valid_sensitivity": valid_sensitivity,
        "valid_specificity": valid_specificity,
        "valid_accuracy": valid_accuracy
    }


def _run(fold, model):
    train_dataset, valid_dataset = get_train_val(fold)

    #     train_sampler = torch.utils.data.Sampler(
    #         train_dataset,
    #         num_replicas=torch.cuda.device_count(),
    #         rank=0,
    #         shuffle=True,
    #     )

    #     valid_sampler = torch.utils.data.Sampler(
    #         valid_dataset,
    #         num_replicas=torch.cuda.device_count(),
    #         rank=0,
    #         shuffle=False,
    #     )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        #         sampler=train_sampler,
        drop_last=True,
        #         num_workers=torch.cuda.device_count(),
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        #         sampler=valid_sampler,
        drop_last=True,
        #         num_workers=torch.cuda.device_count(),
    )

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     device = xm.xla_device()
    model.to(device)

    lr = LR
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"INITIALIZING TRAINING ON {torch.cuda.device_count()} GPU CORES")
    start_time = datetime.now()
    print(f"Start Time: {start_time}")

    logs = fit_gpu(
        model=model,
        epochs=N_EPOCHS,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )

    print(f"Execution time: {datetime.now() - start_time}")

    print("Saving Model")
    torch.save(
        model.state_dict(),
        f'weights/checkpoints/model_efficientnet_b3_{datetime.now().strftime("%Y%m%d-%H%M")}.pth',
    )
    return logs


if __name__ == '__main__':
    seed_everything(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))

    data2019 = pd.read_csv(DATA2019_PATH)
    data2020 = pd.read_csv(DATA2020_PATH)
    data_csv = pd.concat([data2019, data2020]).reset_index()
    
    df_ground_truth, known_category_names = load_isic_training_data(
                    image_folder2019=TRAIN_IMG2019_PATH, image_folder2020=TRAIN_IMG2020_PATH, 
                    labels2019=DATA2019_PATH, labels2020=DATA2020_PATH
    )
    
    df_ground_truth["kfold"] = -1
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for f, (t_, v_) in enumerate(
        kf.split(X=df_ground_truth, y=df_ground_truth["category"])
    ):
        df_ground_truth.loc[v_, "kfold"] = f

    df_fold0 = df_ground_truth[df_ground_truth["kfold"] == 0]
    df_fold1 = df_ground_truth[df_ground_truth["kfold"] == 1]
    df_fold2 = df_ground_truth[df_ground_truth["kfold"] == 2]
    df_fold3 = df_ground_truth[df_ground_truth["kfold"] == 3]
    df_fold4 = df_ground_truth[df_ground_truth["kfold"] == 4]

    print("\n FOLD 0 DISTRIBUTION \n")
    compute_class_dist(df_fold0, known_category_names)
    print("FOLD 1 DISTRIBUTION \n")
    compute_class_dist(df_fold1, known_category_names)
    print("\n FOLD 2 DISTRIBUTION \n")
    compute_class_dist(df_fold2, known_category_names)
    print("\n FOLD 3 DISTRIBUTION \n")
    compute_class_dist(df_fold3, known_category_names)
    print("\n FOLD 4 DISTRIBUTION \n")
    compute_class_dist(df_fold4, known_category_names)

    writer = SummaryWriter()

    model = Model('efficientvit_m5.r224_in1k',pretrained=True)

    for i in range(5):
        start_time = time.time()
        _run(i, model)



