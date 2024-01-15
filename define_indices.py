import pickle
import re
import numpy as np
import pandas as pd
import random
from itertools import permutations 

REMOVED_INDICES = [17257, 23523, 1609, 8458, 8535, 25291, 23971, 19824, 24269, 18154, 23511, 14954, 14568, 13745, 24712,
 18476, 21588, 20376, 22221, 20866, 24558, 17816, 25279, 17535, 24431, 20931, 22229, 18479, 18295, 24581,
 22164, 18627, 22720, 19282, 25177, 25229, 23034, 19012, 23191, 19699, 19664, 23954, 24430, 20807, 24787,
 25161, 20520, 21751, 21988, 24025, 23919, 24281]


def set_ham_inds(mdlParams):
    '''
    Find and set only ham10000 indices in data
    '''
    # Create indices list with HAM10000 only
    mdlParams['HAM10000_inds'] = []
    HAM_START = 24306
    HAM_END = 34320
    for j in range(len(mdlParams['key_list'])):
        try:
            curr_id = [int(s) for s in re.findall(r'\d+',mdlParams['key_list'][j])][-1]
        except:
            continue
        if curr_id >= HAM_START and curr_id <= HAM_END:
            mdlParams['HAM10000_inds'].append(j)
    mdlParams['HAM10000_inds'] = np.array(mdlParams['HAM10000_inds'])
    print("Len ham",len(mdlParams['HAM10000_inds']))
    return mdlParams


def define_cv_indices(mdlParams, exclude_list):
    # with open(mdlParams['saveDir'] + 'indices_isic2019.pkl','rb') as f:
    #     indices = pickle.load(f)
    # mdlParams['trainIndCV'] = indices['trainIndCV']
    # mdlParams['valIndCV'] = indices['valIndCV']
    # print(mdlParams['exclude_inds'])

    # exclude_list = np.array(exclude_list)
    # all_inds = np.arange(len(mdlParams['im_paths']))
    # exclude_inds = all_inds[exclude_list.astype(bool)]
    # for i in range(len(mdlParams['trainIndCV'])):
    #     print('Length before: ', len(mdlParams['trainIndCV'][i]))
    #     mdlParams['trainIndCV'][i] = np.setdiff1d(mdlParams['trainIndCV'][i], REMOVED_INDICES)
    #     print('Length after: ', len(mdlParams['trainIndCV'][i]))
    # for i in range(len(mdlParams['valIndCV'])):
    #     mdlParams['valIndCV'][i] = np.setdiff1d(mdlParams['valIndCV'][i], REMOVED_INDICES)

    labels = pd.read_csv('isic2019/labels/official/binary_labels2019.csv')
    indices = list(labels.index)

    random.shuffle(indices)

    train_ind, val_ind = indices[int(len(indices)*0.2):], indices[:int(len(indices)*0.2)]

    mdlParams['trainIndCV'] = [np.array(train_ind)]
    mdlParams['valIndCV'] = [np.array(val_ind)] 

    
    return mdlParams

