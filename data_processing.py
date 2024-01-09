import imagesize
import scipy
from glob import glob
import numpy as np
import pickle

from model_configs import set_model_config
from create_labels import create_labels_dict
from prepare_files import set_images_dirs, get_preload_imgs
from define_indices import define_cv_indices, set_ham_inds

def set_imgs_means(mdlParams):
    '''
    Set imgs means in params dict
    '''
    mdlParams['images_means'] = np.zeros([len(mdlParams['im_paths']),3])
    for i in range(len(mdlParams['im_paths'])):
        x = scipy.ndimage.imread(mdlParams['im_paths'][i])
        x = x.astype(np.float32)
        # Scale to 0-1
        min_x = np.min(x)
        max_x = np.max(x)
        x = (x-min_x)/(max_x-min_x)
        mdlParams['images_means'][i,:] = np.mean(x,(0,1))
        if i%1000 == 0:
            print(i+1,"images processed for mean...")


def make_crops(mdlParams):
    # Crop positions, always choose multiCropEval to be 4, 9, 16, 25, etc.
    mdlParams['cropPositions'] = np.zeros([len(mdlParams['im_paths']),mdlParams['multiCropEval'],2],dtype=np.int64)
    #mdlParams['imSizes'] = np.zeros([len(mdlParams['im_paths']),mdlParams['multiCropEval'],2],dtype=np.int64)
    for u in range(len(mdlParams['im_paths'])):
        height, width = imagesize.get(mdlParams['im_paths'][u])
        if width < mdlParams['input_size'][0]:
            height = int(mdlParams['input_size'][0]/float(width))*height
            width = mdlParams['input_size'][0]
        if height < mdlParams['input_size'][0]:
            width = int(mdlParams['input_size'][0]/float(height))*width
            height = mdlParams['input_size'][0]
        ind = 0
        for i in range(np.int32(np.sqrt(mdlParams['multiCropEval']))):
            for j in range(np.int32(np.sqrt(mdlParams['multiCropEval']))):
                mdlParams['cropPositions'][u,ind,0] = mdlParams['input_size'][0]/2+i*((width-mdlParams['input_size'][1])/(np.sqrt(mdlParams['multiCropEval'])-1))
                mdlParams['cropPositions'][u,ind,1] = mdlParams['input_size'][1]/2+j*((height-mdlParams['input_size'][0])/(np.sqrt(mdlParams['multiCropEval'])-1))
                #mdlParams['imSizes'][u,ind,0] = curr_im_size[0]

                ind += 1
    check_images(mdlParams)
    return mdlParams


def check_images(mdlParams):
    # Test image sizes
    height = mdlParams['input_size'][0]
    width = mdlParams['input_size'][1]
    for u in range(len(mdlParams['im_paths'])):
        height_test, width_test = imagesize.get(mdlParams['im_paths'][u])
        if width_test < mdlParams['input_size'][0]:
            height_test = int(mdlParams['input_size'][0]/float(width_test))*height_test
            width_test = mdlParams['input_size'][0]
        if height_test < mdlParams['input_size'][0]:
            width_test = int(mdlParams['input_size'][0]/float(height_test))*width_test
            height_test = mdlParams['input_size'][0]
        test_im = np.zeros([width_test,height_test])
        for i in range(mdlParams['multiCropEval']):
            im_crop = test_im[np.int32(mdlParams['cropPositions'][u,i,0]-height/2):np.int32(mdlParams['cropPositions'][u,i,0]-height/2)+height,np.int32(mdlParams['cropPositions'][u,i,1]-width/2):np.int32(mdlParams['cropPositions'][u,i,1]-width/2)+width]
            if im_crop.shape[0] != mdlParams['input_size'][0]:
                print("Wrong shape",im_crop.shape[0],mdlParams['im_paths'][u])
            if im_crop.shape[1] != mdlParams['input_size'][1]:
                print("Wrong shape",im_crop.shape[1],mdlParams['im_paths'][u])


def concat_multisets(mdlParams):
    if len(mdlParams['dataset_names']) > 1:
        restInds = np.array(np.arange(25331,mdlParams['labels_array'].shape[0]))
        for i in range(mdlParams['numCV']):
            mdlParams['trainIndCV'][i] = np.concatenate((mdlParams['trainIndCV'][i],restInds))
    print("Train")
    for i in range(len(mdlParams['trainIndCV'])):
        print(mdlParams['trainIndCV'][i].shape)
    print("Val")
    for i in range(len(mdlParams['valIndCV'])):
        print(mdlParams['valIndCV'][i].shape)

    return mdlParams


def preprocess_data():
    # Empty dict to store machine specific info
    mdlParams = set_model_config()
    mdlParams = create_labels_dict(mdlParams)

    # Define the sets
    path1 = mdlParams['dataDir'] + '/images/'
    print('Images path: ', path1)
    # All sets
    allSets = sorted(glob(path1 + '*/'))

    # allSets = make_official_dataset_first(allSets, mdlParams)

    # Set of keys, for marking old HAM10000
    indices_exclude = None
    mdlParams['key_list'] = []
    if mdlParams['exclude_inds']:
        with open(mdlParams['saveDir'] + 'indices_exclude.pkl','rb') as f:
            indices_exclude = pickle.load(f)

    exclude_list = []

    allSets, mdlParams, exclude_list = set_images_dirs(allSets, mdlParams, exclude_list, indices_exclude)

    # Convert label list to array
    mdlParams['labels_array'] = np.array(mdlParams['labels_list'])
    print('Mean: ', np.mean(mdlParams['labels_array'],axis=0))


    # Perhaps preload images
    if mdlParams['preload']:
        mdlParams = get_preload_imgs(mdlParams)

    # Set imgs means
    if mdlParams['subtract_set_mean']:
        mdlParams = set_imgs_means(mdlParams)

    ### Define Indices ###
    mdlParams = define_cv_indices(mdlParams, exclude_list)

    # Consider case with more than one set
    mdlParams = concat_multisets(mdlParams)

    # Use this for ordered multi crops
    if mdlParams['orderedCrop']:
        mdlParams = make_crops(mdlParams)
    return mdlParams, allSets, exclude_list

