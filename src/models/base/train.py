import torch
from torch import nn
from datetime import datetime
import numpy as np
from lion_pytorch import Lion
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .loss import FocalLoss
from .utils import seed_everything
from .load_data import get_train_val, get_kfold
from .model import Model
from .scheduler import GradualWarmupSchedulerV2
from .constants import LR, BATCH_SIZE, ALPHA2019, ALPHA_MERG, BALANCED_ALPHA, GAMMA, WEIGHT_DECAY, REMOVED_ALPHA, \
                    N_EPOCHS, RANDOM_SEED, EPOCH_WEIGHTS_PATH, MODEL_WEIGHTS_PATH, T0, T_MULT, ETA_MIN, LAST_EPOCH, \
                    MERGED_TASK, REMOVED_TASK, BALANCED_TASK, ISIC2019_TASK, BAD_TASK_ERROR, CUDA_DEVICE, CPU_DEVICE


def fit_gpu(model_name: str,
            model: Model, 
            epochs: int, 
            device: torch.device, 
            criterion: nn.Module, 
            optimizer: nn.Module, 
            writer: SummaryWriter,
            train_loader: torch.utils.data.DataLoader, 
            valid_loader: torch.utils.data.DataLoader = None,
            scheduler: nn.Module = None):

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
                EPOCH_WEIGHTS_PATH % {'model_name': model_name, 'epoch': epoch, 
                                      'datetime': datetime.now().strftime("%Y%m%d-%H%M")},
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


def _run(task: str, model_name: str, fold: int, model: Model):
    if task == ISIC2019_TASK:
        alpha = ALPHA2019
    elif task == BALANCED_TASK:
        alpha = BALANCED_ALPHA
    elif task == REMOVED_TASK:
        alpha = REMOVED_ALPHA
    elif task == MERGED_TASK:
        alpha = ALPHA_MERG
    else:
        raise Exception(BAD_TASK_ERROR)


    seed_everything(RANDOM_SEED)
    labels = get_kfold(task)
    train_dataset, valid_dataset = get_train_val(labels, fold)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE, shuffle=True,
                                               drop_last=True, num_workers=2)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=BATCH_SIZE, shuffle=True,
                                               drop_last=True, num_workers=2)

    device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)

    criterion = FocalLoss(alpha=alpha, gamma=GAMMA, device=device)
    print('Device: ', device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)   
    # optimizer = Lion(model.parameters(), lr=LR) 
    #scheduler = StepLR(optimizer=optimizer, step_size=5, gamma=0.05)

    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=T_MULT, eta_min=ETA_MIN, last_epoch=LAST_EPOCH)

    scheduler_cosine = CosineAnnealingWarmRestarts(optimizer, N_EPOCHS - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    writer = SummaryWriter(comment=model_name)

    print(f"INITIALIZING TRAINING ON {torch.cuda.device_count()} GPU CORES")
    start_time = datetime.now()
    print(f"Start Time: {start_time}")

    logs = fit_gpu(model_name=model_name,
                   model=model,
                   epochs=N_EPOCHS,
                   device=device,
                   criterion=criterion,
                   optimizer=optimizer,
                   writer=writer,
                   train_loader=train_loader,
                   valid_loader=valid_loader,
                   scheduler=scheduler_warmup)

    print(f"Execution time: {datetime.now() - start_time}")

    print("Saving Model")
    torch.save(model.state_dict(),
               MODEL_WEIGHTS_PATH % {'model_name': model_name, 
                                     'datetime': datetime.now().strftime("%Y%m%d-%H%M")})
    return logs
