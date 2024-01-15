import numpy as np

def set_model_config():
    mdlParams = {}
    # Save summaries and model here
    mdlParams['saveDir'] = '/home/konovalyuk/data/isic/'
    print('Save path: ', mdlParams['saveDir'])
    # Data is loaded from here
    mdlParams['dataDir'] = '/home/konovalyuk/isic2019'
    print('Load dataset path: ', mdlParams['dataDir'])

    ### Model Selection ###
    mdlParams['model_type'] = 'efficientnet-b4'
    mdlParams['dataset_names'] = ['official']#,'sevenpoint_rez3_ll']
    mdlParams['file_ending'] = '.jpg'
    mdlParams['exclude_inds'] = False
    mdlParams['same_sized_crops'] = True
    mdlParams['multiCropEval'] = 9
    mdlParams['var_im_size'] = True
    mdlParams['orderedCrop'] = True
    mdlParams['voting_scheme'] = 'average'
    mdlParams['classification'] = True
    mdlParams['balance_classes'] = 2
    mdlParams['extra_fac'] = 1.0
    mdlParams['numClasses'] = 2
    mdlParams['no_c9_eval'] = True
    mdlParams['numOut'] = mdlParams['numClasses']
    mdlParams['numCV'] = 1
    mdlParams['trans_norm_first'] = True
    # Scale up for b1-b7
    mdlParams['input_size'] = [224,224,3]

    ### Training Parameters ###
    # Batch size
    mdlParams['batchSize'] = 20#*len(mdlParams['numGPUs'])
    # Initial learning rate
    mdlParams['learning_rate'] = 0.000015#*len(mdlParams['numGPUs'])
    # Lower learning rate after no improvement over 100 epochs
    mdlParams['lowerLRAfter'] = 25
    # If there is no validation set, start lowering the LR after X steps
    mdlParams['lowerLRat'] = 50
    # Divide learning rate by this value
    mdlParams['LRstep'] = 5
    # Maximum number of training iterations
    mdlParams['training_steps'] = 60 #250
    # Display error every X steps
    mdlParams['display_step'] = 10
    # Scale?
    mdlParams['scale_targets'] = False
    # Peak at test error during training? (generally, dont do this!)
    mdlParams['peak_at_testerr'] = False
    # Print trainerr
    mdlParams['print_trainerr'] = False
    # Subtract trainset mean?
    mdlParams['subtract_set_mean'] = False
    mdlParams['setMean'] = np.array([0.0, 0.0, 0.0])
    mdlParams['setStd'] = np.array([1.0, 1.0, 1.0])

    # Data AUG
    #mdlParams['full_color_distort'] = True
    mdlParams['autoaugment'] = False
    mdlParams['flip_lr_ud'] = True
    mdlParams['full_rot'] = 180
    mdlParams['scale'] = (0.8,1.2)
    mdlParams['shear'] = 10
    mdlParams['cutout'] = 16

    ### Data ###
    mdlParams['preload'] = False
    # Labels first
    # Targets, as dictionary, indexed by im file name
    mdlParams['labels_dict'] = {}

    mdlParams['pathBase'] = ''

    # Save all im paths here
    mdlParams['im_paths'] = []
    mdlParams['labels_list'] = []


    return mdlParams
