import pickle
import re
import numpy as np

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
    with open(mdlParams['saveDir'] + 'indices_isic2019.pkl','rb') as f:
        indices = pickle.load(f)
    mdlParams['trainIndCV'] = indices['trainIndCV']
    mdlParams['valIndCV'] = indices['valIndCV']
    if mdlParams['exclude_inds']:
        exclude_list = np.array(exclude_list)
        all_inds = np.arange(len(mdlParams['im_paths']))
        exclude_inds = all_inds[exclude_list.astype(bool)]
        for i in range(len(mdlParams['trainIndCV'])):
            mdlParams['trainIndCV'][i] = np.setdiff1d(mdlParams['trainIndCV'][i],exclude_inds)
        for i in range(len(mdlParams['valIndCV'])):
            mdlParams['valIndCV'][i] = np.setdiff1d(mdlParams['valIndCV'][i],exclude_inds)

    return mdlParams

