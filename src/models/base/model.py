import timm
import torch
import numpy as np
import pandas as pd
from torch import nn
from datetime import datetime
from torchmetrics.functional import f1_score, confusion_matrix

from .constants import BETA, NUM_CLASSES, CUDA_DEVICE, CPU_DEVICE


class Model(nn.Module):
    model: timm.models.efficientnet.EfficientNet
    num_classes: int

    def __init__(self, timm_model_name: str, n_classes: int = NUM_CLASSES, pretrained: bool = True):
        super(Model, self).__init__()
        self.num_classes = n_classes
        
        self.model = timm.create_model(
            timm_model_name,
            pretrained=pretrained,
            num_classes=self.num_classes,
        )
        
        # self.model.head = nn.Linear(self.model.head, n_classes)
    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = torch.softmax(x, dim=1)
        return x

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader,
                        criterion: nn.Module, optimizer: nn.Module, device: torch.device):
        # keep track of training loss
        epoch_loss = 0.0
        epoch_w_f1 = 0.0

        ###################
        # train the model #
        ###################
        self.model.train()
        for i, (data, target) in enumerate(train_loader):
            # move tensors to GPU if CUDA is available
            if device.type == CUDA_DEVICE:
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

            epoch_loss += loss
            epoch_w_f1 += w_f1

            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)

            # perform a single optimization step (parameter update)            
            optimizer.step()
            if i % 20 == 0:
                print(f"\tBATCH {i+1}/{len(train_loader)} - LOSS: {loss}")
                # print("Accuracy: ", acc_computed)
                    
        return epoch_loss / len(train_loader), epoch_w_f1 / len(train_loader)

    def validate_one_epoch(self, valid_loader: torch.utils.data.DataLoader,
                          criterion: nn.Module, device: torch.device, beta: float = BETA):
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
            if device.type == CUDA_DEVICE:
                data, target = data.cuda(), target.cuda()
            else:
                print(f"{device.type} is your device")                

            with torch.no_grad():
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.forward(data)

                # calculate the batch loss
                loss = criterion(output, target)
                # Calculate Weighted F1
                w_f1 = f1_score(output, 
                                target, 
                                num_classes=self.num_classes, 
                                average="weighted", 
                                task="multiclass"
)
                output = output.to(CPU_DEVICE)
                target = target.to(CPU_DEVICE)

                matrix = confusion_matrix(target=target, preds=np.argmax(output, 1), task='binary', num_classes=self.num_classes)
                tn, fp, fn, tp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    
                sensitivity += tp/(tp+fn + beta)
                specificity += tn/(tn+fp + beta)
                acc_computed += (tp+tn)/(tn+fp+fn+tp)

                # update average validation loss and accuracy
                valid_loss += loss
                valid_w_f1 += w_f1
                

        
        return valid_loss / len(valid_loader), valid_w_f1 / len(valid_loader), sensitivity / len(valid_loader), \
                specificity / len(valid_loader), acc_computed / len(valid_loader)    
