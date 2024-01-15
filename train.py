import os
import pickle
import time
import numpy as np
import scipy
import torch
from pathlib import Path
import importlib
from torchvision import models
import sklearn
import torch.nn as nn
import math
from tqdm import tqdm

from count_loss import getErrClassification_mgpu
from pretrain import set_visible_devices, setup_dataloaders, check_train_params, load_checkpoint
from prepare_files import set_images_dirs
from model_configs import set_model_config
from setup_train import setup_loss_and_optimizer, setup_meta_training
from pretrain import check_trained
from model import define_model, balance_classes
from data_processing import preprocess_data


def run_train(mdlParams, modelVars, allData, save_dict, save_dict_train, eval_set, start_epoch, cv):
    start_time = time.time()
    
    print("Start training...")
    for step in range(start_epoch, mdlParams['training_steps']+1):
        # One Epoch of training
        if step >= mdlParams['lowerLRat']-mdlParams['lowerLRAfter']:
            modelVars['scheduler'].step()
        modelVars['model'].train()
        with tqdm(modelVars['dataloader_trainInd']) as tepoch:
            for j, (inputs, labels, indices) in enumerate(tepoch):
                #print(indices)
                #t_load = time.time()
                # Run optimization
                if mdlParams.get('meta_features',None) is not None:
                    inputs[0] = inputs[0].cuda()
                    inputs[1] = inputs[1].cuda()
                else:
                    inputs = inputs.cuda()
                #print(inputs.shape)
                labels = labels.cuda()
                # zero the parameter gradients
                modelVars['optimizer'].zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    if mdlParams.get('aux_classifier',False):
                        outputs, outputs_aux = modelVars['model'](inputs)
                        loss1 = modelVars['criterion'](outputs, labels)
                        labels_aux = labels.repeat(mdlParams['multiCropTrain'])
                        loss2 = modelVars['criterion'](outputs_aux, labels_aux)
                        loss = loss1 + mdlParams['aux_classifier_loss_fac']*loss2
                    else:
                        #print("load",time.time()-t_load)
                        #t_fwd = time.time()
                        outputs = modelVars['model'](inputs)
                        #print("forward",time.time()-t_fwd)
                        #t_bwd = time.time()
                        loss = modelVars['criterion'](outputs, labels)
                    # Perhaps adjust weighting of the loss by the specific index
                    if mdlParams['balance_classes'] == 6 or mdlParams['balance_classes'] == 7 or mdlParams['balance_classes'] == 8:
                        #loss = loss.cpu()
                        indices = indices.numpy()
                        loss = loss*torch.cuda.FloatTensor(mdlParams['loss_fac_per_example'][indices].astype(np.float32))
                        loss = torch.mean(loss)
                        #loss = loss.cuda()
                    # backward + optimize only if in training phase
                    loss.backward()
                    modelVars['optimizer'].step()
                    #print("backward",time.time()-t_bwd)
        if step % mdlParams['display_step'] == 0 or step == 1:
            # Calculate evaluation metrics
            if mdlParams['classification']:
                # Adjust model state
                modelVars['model'].eval()
                # Get metrics
                loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, _ = getErrClassification_mgpu(mdlParams, eval_set, modelVars)
                # Save in mat
                save_dict['loss'].append(loss)
                save_dict['acc'].append(accuracy)
                save_dict['wacc'].append(waccuracy)
                save_dict['auc'].append(auc)
                save_dict['sens'].append(sensitivity)
                save_dict['spec'].append(specificity)
                save_dict['f1'].append(f1)
                save_dict['step_num'].append(step)
                if os.path.isfile(mdlParams['saveDir'] + '/progression_'+eval_set+'.mat'):
                    os.remove(mdlParams['saveDir'] + '/progression_'+eval_set+'.mat')
                scipy.io.savemat(mdlParams['saveDir'] + '/progression_'+eval_set+'.mat',save_dict)
            eval_metric = -np.mean(waccuracy)
            # Check if we have a new best value
            if eval_metric < mdlParams['valBest']:
                mdlParams['valBest'] = eval_metric
                if mdlParams['classification']:
                    allData['f1Best'][cv] = f1
                    allData['sensBest'][cv] = sensitivity
                    allData['specBest'][cv] = specificity
                    allData['accBest'][cv] = accuracy
                    allData['waccBest'][cv] = waccuracy
                    allData['aucBest'][cv] = auc
                oldBestInd = mdlParams['lastBestInd']
                mdlParams['lastBestInd'] = step
                allData['convergeTime'][cv] = step
                # Save best predictions
                allData['bestPred'][cv] = predictions
                allData['targets'][cv] = targets
                # Write to File
                with open(mdlParams['saveDirBase'] + '/CV.pkl', 'wb') as f:
                    pickle.dump(allData, f, pickle.HIGHEST_PROTOCOL)
                # Delte previously best model
                if os.path.isfile(mdlParams['saveDir'] + '/checkpoint_best-' + str(oldBestInd) + '.pt'):
                    os.remove(mdlParams['saveDir'] + '/checkpoint_best-' + str(oldBestInd) + '.pt')
                # Save currently best model
                state = {'epoch': step, 'valBest': mdlParams['valBest'], 'lastBestInd': mdlParams['lastBestInd'], 'state_dict': modelVars['model'].state_dict(),'optimizer': modelVars['optimizer'].state_dict()}
                torch.save(state, mdlParams['saveDir'] + '/checkpoint_best-' + str(step) + '.pt')

            # If its not better, just save it delete the last checkpoint if it is not current best one
            # Save current model
            state = {'epoch': step, 'valBest': mdlParams['valBest'], 'lastBestInd': mdlParams['lastBestInd'], 'state_dict': modelVars['model'].state_dict(),'optimizer': modelVars['optimizer'].state_dict()}
            torch.save(state, mdlParams['saveDir'] + '/checkpoint-' + str(step) + '.pt')
            # Delete last one
            if step == mdlParams['display_step']:
                lastInd = 1
            else:
                lastInd = step-mdlParams['display_step']
            if os.path.isfile(mdlParams['saveDir'] + '/checkpoint-' + str(lastInd) + '.pt'):
                os.remove(mdlParams['saveDir'] + '/checkpoint-' + str(lastInd) + '.pt')
            # Duration so far
            duration = time.time() - start_time

            # Print
            if mdlParams['classification']:
                print("\n")
                print("Config:", '2019.test_effb0_ss')
                print('Fold: %d Epoch: %d/%d (%d h %d m %d s)' % (cv,step,mdlParams['training_steps'], int(duration/3600), int(np.mod(duration,3600)/60), int(np.mod(np.mod(duration,3600),60))) + time.strftime("%d.%m.-%H:%M:%S", time.localtime()))
                print("Loss on ",eval_set,"set: ",loss," Accuracy: ",accuracy," F1: ",f1," (best WACC: ",-mdlParams['valBest']," at Epoch ",mdlParams['lastBestInd'],")")
                print("Auc",auc,"Mean AUC",np.mean(auc))
                print("Per Class Acc",waccuracy,"Weighted Accuracy",np.mean(waccuracy))
                print("Sensitivity: ",sensitivity,"Specificity",specificity)
                print("Confusion Matrix")
                print(conf_matrix)
                # Potentially peek at test error
                if mdlParams['peak_at_testerr']:
                    loss, accuracy, sensitivity, specificity, _, f1, _, _, _, _, _ = getErrClassification_mgpu(mdlParams, 'testInd', modelVars)
                    print("Test loss: ",loss," Accuracy: ",accuracy," F1: ",f1)
                    print("Sensitivity: ",sensitivity,"Specificity",specificity)
                # Potentially print train err
                if mdlParams['print_trainerr'] and 'train' not in eval_set:
                    loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, _ = getErrClassification_mgpu(mdlParams, 'trainInd', modelVars)
                    # Save in mat
                    save_dict_train['loss'].append(loss)
                    save_dict_train['acc'].append(accuracy)
                    save_dict_train['wacc'].append(waccuracy)
                    save_dict_train['auc'].append(auc)
                    save_dict_train['sens'].append(sensitivity)
                    save_dict_train['spec'].append(specificity)
                    save_dict_train['f1'].append(f1)
                    save_dict_train['step_num'].append(step)
                    if os.path.isfile(mdlParams['saveDir'] + '/progression_trainInd.mat'):
                        os.remove(mdlParams['saveDir'] + '/progression_trainInd.mat')
                    scipy.io.savemat(mdlParams['saveDir'] + '/progression_trainInd.mat',save_dict_train)
                    print("Train loss: ",loss," Accuracy: ",accuracy," F1: ",f1)
                    print("Sensitivity: ",sensitivity,"Specificity",specificity)
    return mdlParams, allData, save_dict, save_dict_train

def train():
    mdlParams, allSets, exclude_list = preprocess_data()
    # Indicate training
    mdlParams['trainSetState'] = 'train'

    # Collect model variables
    modelVars = {}

    # Path name from filename
    mdlParams['saveDirBase'] = mdlParams['saveDir'] + '2019.test_effb0_ss'
    mdlParams_model = mdlParams.copy()

    # Set visible devices
    mdlParams = set_visible_devices(mdlParams)

    # Check if there is a validation set, if not, evaluate train error instead
    if 'valIndCV' in mdlParams or 'valInd' in mdlParams:
        eval_set = 'valInd'
        print("Evaluating on validation set during training.")
    else:
        eval_set = 'trainInd'
        print("No validation set, evaluating on training set during training.")

    # Check if there were previous ones that have alreary bin learned
    prevFile = Path(mdlParams['saveDirBase'] + '/CV.pkl')
    #print(prevFile)
    if prevFile.exists():
        print("Part of CV already done")
        with open(mdlParams['saveDirBase'] + '/CV.pkl', 'rb') as f:
            allData = pickle.load(f)
    else:
        allData = {}
        allData['f1Best'] = {}
        allData['sensBest'] = {}
        allData['specBest'] = {}
        allData['accBest'] = {}
        allData['waccBest'] = {}
        allData['aucBest'] = {}
        allData['convergeTime'] = {}
        allData['bestPred'] = {}
        allData['targets'] = {}

    # Take care of CV
    if mdlParams.get('cv_subset',None) is not None:
        cv_set = mdlParams['cv_subset']
    else:
        cv_set = range(mdlParams['numCV'])
    for cv in cv_set:
        # Reset model graph
        importlib.reload(models)
        #importlib.reload(torchvision)


        # Check if this fold was already trained
        mdlParams, already_trained = check_trained(mdlParams, cv)
        if already_trained:
            continue
        print("CV set",cv)

        mdlParams, modelVars, load_old = check_train_params(mdlParams, modelVars, cv)

        # Save training progress in here
        save_dict = {}
        save_dict['acc'] = []
        save_dict['loss'] = []
        save_dict['wacc'] = []
        save_dict['auc'] = []
        save_dict['sens'] = []
        save_dict['spec'] = []
        save_dict['f1'] = []
        save_dict['step_num'] = []

        save_dict_train = {}
        if mdlParams['print_trainerr']:
            save_dict_train['acc'] = []
            save_dict_train['loss'] = []
            save_dict_train['wacc'] = []
            save_dict_train['auc'] = []
            save_dict_train['sens'] = []
            save_dict_train['spec'] = []
            save_dict_train['f1'] = []
            save_dict_train['step_num'] = []

        # Potentially calculate setMean to subtract
        if mdlParams['subtract_set_mean'] == 1:
            mdlParams['setMean'] = np.mean(mdlParams['images_means'][mdlParams['trainInd'],:],(0))
            print("Set Mean",mdlParams['setMean'])

        # balance classes
        mdlParams, class_weights = balance_classes(mdlParams)

        # Meta scaler
        if mdlParams.get('meta_features',None) is not None and mdlParams['scale_features']:
            mdlParams['feature_scaler_meta'] = sklearn.preprocessing.StandardScaler().fit(mdlParams['meta_array'][mdlParams['trainInd'],:])
            print("scaler mean",mdlParams['feature_scaler_meta'].mean_,"var",mdlParams['feature_scaler_meta'].var_)

        # Setup dataloaders
        modelVars = setup_dataloaders(mdlParams, modelVars)

        #print("Setdiff",np.setdiff1d(mdlParams['trainInd'],mdlParams['trainInd']))
        # Define model

        modelVars = define_model(mdlParams, modelVars, cv)

        # Take care of meta case

        modelVars = setup_meta_training(mdlParams, modelVars)

        print(mdlParams['numGPUs'])
        # multi gpu support
        if len(mdlParams['numGPUs']) > 1:
            modelVars['model'] = nn.DataParallel(modelVars['model'])
        modelVars['model'] = modelVars['model'].cuda()

        #summary(modelVars['model'], modelVars['model'].input_size)# (mdlParams['input_size'][2], mdlParams['input_size'][0], mdlParams['input_size'][1]))
        # Loss, with class weighting

        modelVars = setup_loss_and_optimizer(mdlParams, modelVars, class_weights)

        # Set up training
        # loading from checkpoint
        mdlParams, start_epoch = load_checkpoint(mdlParams, modelVars, load_old)

        # Num batches
        numBatchesTrain = int(math.floor(len(mdlParams['trainInd'])/mdlParams['batchSize']))
        print("Train batches",numBatchesTrain)

        # Run training

        mdlParams, allData, save_dict, save_dict_train = run_train(mdlParams, modelVars, allData, save_dict, save_dict_train, eval_set, start_epoch, cv)

        # Free everything in modelvars
        modelVars.clear()
        # After CV Training: print CV results and save them
        print("Best F1:",allData['f1Best'][cv])
        print("Best Sens:",allData['sensBest'][cv])
        print("Best Spec:",allData['specBest'][cv])
        print("Best Acc:",allData['accBest'][cv])
        print("Best Per Class Accuracy:",allData['waccBest'][cv])
        print("Best Weighted Acc:",np.mean(allData['waccBest'][cv]))
        print("Best AUC:",allData['aucBest'][cv])
        print("Best Mean AUC:",np.mean(allData['aucBest'][cv]))
        print("Convergence Steps:",allData['convergeTime'][cv])
        with open('train_logs.txt', 'w+') as file:
            file.write(f'''Best F1: {allData['f1Best'][cv]}\n
                    Best Sens: {allData['sensBest'][cv]}\n
                    Best Acc: {allData['accBest'][cv]}\n
                    Best Per Class Accuracy: {allData['waccBest'][cv]}\n
                    Best Weighted Acc: {np.mean(allData['waccBest'][cv])}\n
                    Best AUC: {allData['aucBest'][cv]}\n
                    Best Mean AUC: {np.mean(allData['aucBest'][cv])}\n
                    Convergence Steps: {allData['convergeTime'][cv]}''')


if __name__ == '__main__':
    train()
