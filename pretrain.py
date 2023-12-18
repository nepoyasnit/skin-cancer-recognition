import re
import os
from glob import glob
import torch
import psutil
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import ISICDataset
from process_image import StratifiedSampler



def set_visible_devices(mdlParams):
    mdlParams['numGPUs']= [[int(s) for s in re.findall(r'\d+','gpu0')][-1]]
    cuda_str = ""
    for i in range(len(mdlParams['numGPUs'])):
        cuda_str = cuda_str + str(mdlParams['numGPUs'][i])
        if i is not len(mdlParams['numGPUs'])-1:
            cuda_str = cuda_str + ","
    print("Devices to use:",cuda_str)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str
    return mdlParams

def check_trained(mdlParams, cv):
    already_trained = False
    if 'valIndCV' in mdlParams:
        mdlParams['saveDir'] = mdlParams['saveDirBase'] + '/CVSet' + str(cv)
    if os.path.isdir(mdlParams['saveDirBase']):
        if os.path.isdir(mdlParams['saveDir']):
            all_max_iter = []
            for name in os.listdir(mdlParams['saveDir']):
                int_list = [int(s) for s in re.findall(r'\d+',name)]
                if len(int_list) > 0:
                    all_max_iter.append(int_list[-1])
                #if '-' + str(mdlParams['training_steps'])+ '.pt' in name:
                #    print("Fold %d already fully trained"%(cv))
                #    already_trained = True
            all_max_iter = np.array(all_max_iter)
            if len(all_max_iter) > 0 and np.max(all_max_iter) >= mdlParams['training_steps']:
                print("Fold %d already fully trained with %d iterations"%(cv,np.max(all_max_iter)))
                already_trained = True

    return mdlParams, already_trained

def check_train_params(mdlParams, modelVars, cv):
    #print("here")
    modelVars['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Def current CV set
    mdlParams['trainInd'] = mdlParams['trainIndCV'][cv]
    if 'valIndCV' in mdlParams:
        mdlParams['valInd'] = mdlParams['valIndCV'][cv]
    # Def current path for saving stuff
    if 'valIndCV' in mdlParams:
        mdlParams['saveDir'] = mdlParams['saveDirBase'] + '/CVSet' + str(cv)
    else:
        mdlParams['saveDir'] = mdlParams['saveDirBase']
    # Create basepath if it doesnt exist yet
    if not os.path.isdir(mdlParams['saveDirBase']):
        os.mkdir(mdlParams['saveDirBase'])
    # Check if there is something to load
    load_old = 0
    if os.path.isdir(mdlParams['saveDir']):
        # Check if a checkpoint is in there
        if len([name for name in os.listdir(mdlParams['saveDir'])]) > 0:
            load_old = 1
            print("Loading old model")
        else:
            # Delete whatever is in there (nothing happens)
            filelist = [os.remove(mdlParams['saveDir'] +'/'+f) for f in os.listdir(mdlParams['saveDir'])]
    else:
        os.mkdir(mdlParams['saveDir'])

    return mdlParams, modelVars, load_old

def setup_dataloaders(mdlParams,  modelVars):
    num_workers = psutil.cpu_count(logical=False)
    # For train
    dataset_train = ISICDataset(mdlParams, 'trainInd')
    # For val
    dataset_val = ISICDataset(mdlParams, 'valInd')
    if mdlParams['multiCropEval'] > 0:
        modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['multiCropEval'], shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['batchSize'], shuffle=False, num_workers=num_workers, pin_memory=True)

    if mdlParams['balance_classes'] == 12 or mdlParams['balance_classes'] == 13:
        #print(np.argmax(mdlParams['labels_array'][mdlParams['trainInd'],:],1).size(0))
        strat_sampler = StratifiedSampler(mdlParams)
        modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], sampler=strat_sampler, num_workers=num_workers, pin_memory=True)
    else:
        modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    return modelVars

def load_checkpoint(mdlParams, modelVars, load_old):
    if load_old:
        # Find last, not last best checkpoint
        files = glob(mdlParams['saveDir']+'/*')
        global_steps = np.zeros([len(files)])
        for i in range(len(files)):
            # Use meta files to find the highest index
            if 'best' in files[i]:
                continue
            if 'checkpoint-' not in files[i]:
                continue
            # Extract global step
            nums = [int(s) for s in re.findall(r'\d+',files[i])]
            global_steps[i] = nums[-1]
        # Create path with maximum global step found
        chkPath = mdlParams['saveDir'] + '/checkpoint-' + str(int(np.max(global_steps))) + '.pt'
        print("Restoring: ",chkPath)
        # Load
        state = torch.load(chkPath)
        # Initialize model and optimizer
        modelVars['model'].load_state_dict(state['state_dict'])
        modelVars['optimizer'].load_state_dict(state['optimizer'])
        start_epoch = state['epoch']+1
        mdlParams['valBest'] = state.get('valBest',1000)
        mdlParams['lastBestInd'] = state.get('lastBestInd',int(np.max(global_steps)))
    else:
        start_epoch = 1
        mdlParams['lastBestInd'] = -1
        # Track metrics for saving best model
        mdlParams['valBest'] = 1000

    return mdlParams, start_epoch
