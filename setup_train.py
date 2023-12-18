import numpy as np
import torch
from torchvision import models
from count_loss import FocalLoss
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from lion_pytorch import Lion


def setup_meta_training(mdlParams, modelVars):
    if mdlParams.get('meta_features',None) is not None:
        # freeze cnn first
        if mdlParams['freeze_cnn']:
            # deactivate all
            for param in modelVars['model'].parameters():
                param.requires_grad = False
            if 'efficient' in mdlParams['model_type']:
                # Activate fc
                for param in modelVars['model']._fc.parameters():
                    param.requires_grad = True
            elif 'wsl' in mdlParams['model_type']:
                # Activate fc
                for param in modelVars['model'].fc.parameters():
                    param.requires_grad = True
            else:
                # Activate fc
                for param in modelVars['model'].last_linear.parameters():
                    param.requires_grad = True
        else:
            # mark cnn parameters
            for param in modelVars['model'].parameters():
                param.is_cnn_param = True
            # unmark fc
            for param in modelVars['model']._fc.parameters():
                param.is_cnn_param = False
        # modify model
        modelVars['model'] = models.modify_meta(mdlParams,modelVars['model'])
        # Mark new parameters
        for param in modelVars['model'].parameters():
            if not hasattr(param, 'is_cnn_param'):
                param.is_cnn_param = False
    return modelVars

def setup_loss_and_optimizer(mdlParams, modelVars, class_weights):
    if mdlParams.get('focal_loss',False):
        modelVars['criterion'] = FocalLoss(alpha=class_weights.tolist())
    elif mdlParams['balance_classes'] == 3 or mdlParams['balance_classes'] == 0 or mdlParams['balance_classes'] == 12:
        modelVars['criterion'] = nn.CrossEntropyLoss()
    elif mdlParams['balance_classes'] == 8:
        modelVars['criterion'] = nn.CrossEntropyLoss(reduce=False)
    elif mdlParams['balance_classes'] == 6 or mdlParams['balance_classes'] == 7:
        modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.tensor(class_weights.astype(np.float32), dtype=torch.float, device='cuda'), reduce=False)
    elif mdlParams['balance_classes'] == 10:
        modelVars['criterion'] = FocalLoss(mdlParams['numClasses'])
    elif mdlParams['balance_classes'] == 11:
        modelVars['criterion'] = FocalLoss(mdlParams['numClasses'],alpha=torch.tensor(class_weights.astype(np.float32), dtype=torch.float, device='cuda'))
    else:
        modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.tensor(class_weights.astype(np.float32), dtype=torch.float, device='cuda'))
    if mdlParams.get('meta_features') is not None:
        if mdlParams['freeze_cnn']:
            modelVars['optimizer'] = Lion(filter(lambda p: p.requires_grad, modelVars['model'].parameters()), lr=mdlParams['learning_rate_meta'])
            print(f'################## {mdlParams["freeze_cnn"]} ############################')
            # sanity check
            for param in filter(lambda p: p.requires_grad, modelVars['model'].parameters()):
                print(param.name,param.shape)
        else:
            modelVars['optimizer'] = Lion(filter(lambda p: p.requires_grad, modelVars['model'].parameters()), lr=mdlParams['learning_rate_meta'])
    else:
            modelVars['optimizer'] = Lion(filter(lambda p: p.requires_grad, modelVars['model'].parameters()), lr=mdlParams['learning_rate'])

    # Decay LR by a factor of 0.1 every 7 epochs
    modelVars['scheduler'] = lr_scheduler.StepLR(modelVars['optimizer'], step_size=mdlParams['lowerLRAfter'], gamma=1/np.float32(mdlParams['LRstep']))

    # Define softmax
    modelVars['softmax'] = nn.Softmax(dim=1)

    return modelVars

