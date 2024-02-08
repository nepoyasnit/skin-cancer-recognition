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
from torch.nn import functional as F
from glob import glob
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch import nn
from torchsummary import summary
import warnings
import time
from datetime import datetime
from torchvision.transforms.functional import pil_to_tensor
from sklearn.utils.class_weight import compute_class_weight

from torch.optim.lr_scheduler import StepLR

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

from torchmetrics.functional import f1_score, confusion_matrix
from timm.scheduler import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter


RANDOM_SEED = 21
IMG_SIZE = 224
BATCH_SIZE = 277 # optimal by formula (gpu_mem - model_size) / (forw_backw_size)
LR = 3e-05
ALPHA = 1 # because of nevus distribution
GAMMA = 2
N_EPOCHS = 20
# [1.2886929  0.81698016]

CORE_PATH = ""
DATA_PATH = "../../isic2019/labels/official/binary_labels_balanced.csv"
TRAIN_IMG_PATH = "../../isic2019/images/official/"

_mean = np.array([0.6237459654304592, 0.5201169854503829, 0.5039494477029685])
_std = np.array([0.24196317678786788, 0.2233599432947672, 0.23118716487089888])

class FocalLoss(nn.Module):
    """
    binary focal loss
    """

    def __init__(self, alpha=ALPHA, gamma=GAMMA):
        super(FocalLoss, self).__init__()
        self.weight = torch.Tensor([alpha, alpha]).cuda()
        self.nllLoss = nn.NLLLoss(weight=self.weight)
        self.gamma = gamma

    def forward(self, input, target):
        softmax = F.softmax(input, dim=1)
        log_logits = torch.log(softmax)
        fix_weights = (1 - softmax) ** self.gamma
        logits = fix_weights * log_logits
        return self.nllLoss(logits, target)


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self,
        image_paths,
        targets,
        resize,
        augmentations=None,
        backend="pil",
        channel_first=True,):
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
            if self.augmentations is not None:
                image = self.augmentations(image)
        else:
            raise Exception("Backend not implemented")
        return image, targets
    

class Model(nn.Module):
    def __init__(self, timm_model_name, n_classes=2, pretrained=False):

        super(Model, self).__init__()
        self.num_classes = n_classes

        self.model = timm.create_model(
            timm_model_name,
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
            else:
                print(f"{device.type} is your device")

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.forward(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Calculate Weighted F1
            w_f1 = f1_score(output, 
                            target, 
                            num_classes=self.num_classes, 
                            average="weighted", 
                            task="multiclass")
            # output = output.to('cpu').detach().numpy()
            # target = target.to('cpu').detach().numpy()

            # tn, fp, fn, tp = confusion_matrix(target, np.argmax(output, 1), labels=[0,1]).ravel()
            # sensitivity = tp/(tp+fn)
            # specificity = tn/(tn+fp)
            # acc_computed = (tp+tn)/(tn+fp+fn+tp)
            # torchmetrics.functional.f1(output,target,num_classes=len(known_category_names),average='weighted')
            # update training loss and accuracy
            epoch_loss += loss
            epoch_w_f1 += w_f1

            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)

            # perform a single optimization step (parameter update)            
            optimizer.step()
            if i % 20 == 0:
                print(f"\tBATCH {i+1}/{len(train_loader)} - LOSS: {loss}")
                # print("Accuracy: ", acc_computed)
                    
        #epoch_loss.to('cpu').detach().numpy()
        #epoch_w_f1.to('cpu').detach().numpy()

        return epoch_loss / len(train_loader), epoch_w_f1 / len(train_loader)

    def validate_one_epoch(self, valid_loader, criterion, device, beta=0.000000001):
        # keep track of validation loss
        valid_loss = 0.0
        valid_w_f1 = 0.0
        sensitivity = 0.0
        specificity = 0.0
        acc_computed = 0.0

        ######################
        # validate the model #
        ######################
        self.model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if device.type == "cuda":
                data, target = data.cuda(), target.cuda()
            else:
                print(f"{device.type} is your device")                

            with torch.no_grad():
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)

                # calculate the batch loss
                loss = criterion(output, target)
                # Calculate Weighted F1
                w_f1 = f1_score(output, 
                                target, 
                                num_classes=self.num_classes, 
                                average="weighted", 
                                task="multiclass"
)
                output = output.to('cpu')
                target = target.to('cpu')


                matrix = confusion_matrix(target=target, preds=np.argmax(output, 1), task='binary', num_classes=2)
                tn, fp, fn, tp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    
                sensitivity += tp/(tp+fn + beta)
                specificity += tn/(tn+fp + beta)
                acc_computed += (tp+tn)/(tn+fp+fn+tp)

                # update average validation loss and accuracy
                valid_loss += loss
                valid_w_f1 += w_f1
                
        #valid_loss = valid_loss.cpu().numpy()
        #valid_w_f1 = valid_w_f1.cpu().numpy()

        
        return valid_loss / len(valid_loader), valid_w_f1 / len(valid_loader), sensitivity / len(valid_loader), \
                specificity / len(valid_loader), acc_computed / len(valid_loader)    

    
def get_transforms(image_size, rgb_mean, rgb_std):
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


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()


def load_isic_training_data(image_folder, ground_truth_file):
    df_ground_truth = pd.read_csv(ground_truth_file)
    # Category names
    known_category_names = list(df_ground_truth.columns.values[1:3])
    
    # Add path and category columns
    df_ground_truth['path'] = df_ground_truth.apply(lambda row : os.path.join(image_folder, row['image_name']+'.jpg'), axis=1)
    df_ground_truth['category'] = np.argmax(np.array(df_ground_truth.iloc[:,1:3]), axis=1)
    return df_ground_truth, known_category_names


def compute_class_dist(df,known_category_names):
    sample_count_train = df.shape[0]
    count_per_category_train = Counter(df['category'])
    for i, c in enumerate(known_category_names):
        print("'%s':\t%d\t(%.2f%%)" % (c, count_per_category_train[i], count_per_category_train[i]*100/sample_count_train))

    return 


def get_train_val(fold, mean, std):
    df_train = df_ground_truth[df_ground_truth.kfold != fold]
    df_valid = df_ground_truth[df_ground_truth.kfold == fold]
    
    train_images = df_train['path'].to_list()
    train_targets = df_train['category'].to_numpy()
    
    valid_images = df_valid['path'].to_list()
    valid_targets = df_valid['category'].to_numpy()
    
    train_aug, valid_aug = get_transforms(IMG_SIZE, mean, std)

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
    

def fit_gpu(model, 
            epochs, 
            device, 
            criterion, 
            optimizer, 
            train_loader, 
            valid_loader=None,
            scheduler=None):

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

        if valid_loader is not None:
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

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min and epoch != 1:
                print(
                    "Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...".format(
                        valid_loss_min, valid_loss
                    )
                )
            torch.save(
                model.state_dict(),
                f'weights/checkpoints/levit2019_balanced/levit256_{epoch}_{datetime.now().strftime("%Y%m%d-%H%M")}.pth',
            )
            valid_loss_min = valid_loss
        if scheduler:
            scheduler.step()
            print('Learning rate: ', scheduler.get_last_lr())
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
    train_dataset, valid_dataset = get_train_val(fold, _mean, _std)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               drop_last=True, num_workers=torch.cuda.device_count())

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=BATCH_SIZE,
                                               drop_last=True, num_workers=torch.cuda.device_count())

    criterion = FocalLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)    
    #scheduler = StepLR(optimizer=optimizer, step_size=5, gamma=0.05)


    print(f"INITIALIZING TRAINING ON {torch.cuda.device_count()} GPU CORES")
    start_time = datetime.now()
    print(f"Start Time: {start_time}")

    logs = fit_gpu(model=model,
                   epochs=N_EPOCHS,
                   device=device,
                   criterion=criterion,
                   optimizer=optimizer,
                   train_loader=train_loader,
                   valid_loader=valid_loader,
                   scheduler=None)

    print(f"Execution time: {datetime.now() - start_time}")

    print("Saving Model")
    torch.save(model.state_dict(),
               f'weights/checkpoints/levit2019_balanced/model-levit256_{datetime.now().strftime("%Y%m%d-%H%M")}.pth',)
    return logs


def evaluate_model(model_name):
    model.load_state_dict(torch.load(model_name))
    dataset = get_whole_dataset()
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=BATCH_SIZE,
                                              drop_last=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = FocalLoss()
    model.to(device)
    
    valid_loss, valid_w_f1, sensitivity, specificity, accuracy = model.validate_one_epoch(data_loader, 
                                                                                          criterion, 
                                                                                          device)
    return valid_loss, valid_w_f1, sensitivity, specificity, accuracy


if __name__ == "__main__":

    seed_everything(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_csv = pd.read_csv(DATA_PATH)
    
    df_ground_truth, known_category_names = load_isic_training_data(TRAIN_IMG_PATH,
                                                                    DATA_PATH)
    
    df_ground_truth["kfold"] = -1
    kf = StratifiedKFold(n_splits=5, 
                         shuffle=True, 
                         random_state=RANDOM_SEED)
    
    for f, (t_, v_) in enumerate(kf.split(X=df_ground_truth, y=df_ground_truth["category"])):
        df_ground_truth.loc[v_, "kfold"] = f

    writer = SummaryWriter(comment='levit2019_balanced')

    model = Model('levit_256.fb_dist_in1k',pretrained=True)
    model.to(device)
    print(compute_class_weight(class_weight='balanced', classes=np.unique(data_csv.nevus), y=data_csv.nevus))

    for i in range(1):
        _run(i, model)
