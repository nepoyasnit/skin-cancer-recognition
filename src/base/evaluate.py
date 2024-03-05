import torch
import numpy as np
from torchmetrics.functional import f1_score, confusion_matrix

from .load_data import load_ph_test_data
from .model import Model
from .loss import FocalLoss
from .constants import TEST_ALPHA, GAMMA, WEIGHTS_PATH, TEST_IMG_PATH, TEST_LABELS_PATH, BETA, NUM_CLASSES


def evaluate_model(model: Model, model_name: str, model_weights: str, 
                   alpha: float = TEST_ALPHA, gamma: float = GAMMA):
    
    model.load_state_dict(torch.load(WEIGHTS_PATH % {'model_name': model_name} + model_weights))
    model.eval()
    test_dataset = load_ph_test_data(TEST_IMG_PATH, TEST_LABELS_PATH)
    
    data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=len(test_dataset))    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    model.to(device)
    
    test_loss, test_w_f1, test_sens, test_spec, test_acc = model.validate_one_epoch(data_loader, 
                                                                                        criterion, 
                                                                                        device)
    return test_loss, test_w_f1, test_sens, test_spec, test_acc


def evaluate_ensemble(models: list[Model], beta=BETA):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loss = 0.0
    test_w_f1 = 0.0
    sensitivity = 0.0
    specificity = 0.0
    acc_computed = 0.0

    for model in models:
        model.to(device)
        model.eval()
    
    test_dataset = load_ph_test_data(TEST_IMG_PATH, TEST_LABELS_PATH)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=len(test_dataset)) 
    criterion = FocalLoss(alpha=TEST_ALPHA, gamma=GAMMA)
   

    for data, target in test_loader:
        if device.type == "cuda":
            data, target = data.cuda(), target.cuda()
        else:
            print(f"{device.type} is your device")
        output = torch.zeros([len(test_dataset), 2]).cuda()
        with torch.no_grad():
            for model in models:
                output += model(data)
            
            output /= len(models)

            loss = criterion(output, target)

            w_f1 = f1_score(output, 
                    target, 
                    num_classes=NUM_CLASSES, 
                    average="weighted", 
                    task="multiclass")
            
            output = output.to('cpu')
            target = target.to('cpu')


            matrix = confusion_matrix(target=target, preds=np.argmax(output, 1), task='binary', num_classes=NUM_CLASSES)
            tn, fp, fn, tp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]

            sensitivity += tp/(tp+fn + beta)
            specificity += tn/(tn+fp + beta)
            acc_computed += (tp+tn)/(tn+fp+fn+tp)

            # update average validation loss and accuracy
            test_loss += loss
            test_w_f1 += w_f1
        
        return test_loss / len(test_loader), test_w_f1 / len(test_loader), sensitivity / len(test_loader), \
                specificity / len(test_loader), acc_computed / len(test_loader)    


