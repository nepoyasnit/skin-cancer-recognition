import glob
import re
from efficientnet_pytorch import EfficientNet
import functools
from collections import OrderedDict
import torch.nn as nn
import types
import torch.nn.functional as F
from sklearn.utils import class_weight


import numpy as np
import torch
import pickle


def efficientnet_b0(config):
    return EfficientNet.from_pretrained('efficientnet-b0',num_classes=config['numClasses'])

def efficientnet_b1(config):
    return EfficientNet.from_pretrained('efficientnet-b1',num_classes=config['numClasses'])

def efficientnet_b2(config):
    return EfficientNet.from_pretrained('efficientnet-b2',num_classes=config['numClasses'])

def efficientnet_b3(config):
    return EfficientNet.from_pretrained('efficientnet-b3',num_classes=config['numClasses'])

def efficientnet_b4(config):
    return EfficientNet.from_pretrained('efficientnet-b4',num_classes=config['numClasses'])

def efficientnet_b5(config):
    return EfficientNet.from_pretrained('efficientnet-b5',num_classes=config['numClasses'])

def efficientnet_b6(config):
    return EfficientNet.from_pretrained('efficientnet-b6',num_classes=config['numClasses'])

def efficientnet_b7(config):
    return EfficientNet.from_pretrained('efficientnet-b7',num_classes=config['numClasses'])

model_map = OrderedDict([('efficientnet-b4', efficientnet_b4),])

def modify_densenet_avg_pool(model):
    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = torch.mean(torch.mean(x,2), 2)
        #x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)

    return model


def getModel(config):
  """Returns a function for a model
  Args:
    config: dictionary, contains configuration
  Returns:
    model: A class that builds the desired model
  Raises:
    ValueError: If model name is not recognized.
  """
  if config['model_type'] in model_map:
    func = model_map[config['model_type'] ]
    @functools.wraps(func)
    def model():
        return func(config)
  else:
      raise ValueError('Name of model unknown %s' % config['model_name'] )
  return model

def define_model(mdlParams, modelVars, cv):
    modelVars['model'] = getModel(mdlParams)()

    # Load trained model
    if mdlParams.get('meta_features',None) is not None:
        # Find best checkpoint
        files = glob(mdlParams['model_load_path'] + '/CVSet' + str(cv) + '/*')
        global_steps = np.zeros([len(files)])
        #print("files",files)
        for i in range(len(files)):
            # Use meta files to find the highest index
            if 'best' not in files[i]:
                continue
            if 'checkpoint' not in files[i]:
                continue
            # Extract global step
            nums = [int(s) for s in re.findall(r'\d+',files[i])]
            global_steps[i] = nums[-1]
        # Create path with maximum global step found
        chkPath = mdlParams['model_load_path'] + '/CVSet' + str(cv) + '/checkpoint_best-' + str(int(np.max(global_steps))) + '.pt'
        print("Restoring lesion-trained CNN for meta data training: ",chkPath)
        # Load
        state = torch.load(chkPath)
        # Initialize model
        curr_model_dict = modelVars['model'].state_dict()
        for name, param in state['state_dict'].items():
            #print(name,param.shape)
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if curr_model_dict[name].shape == param.shape:
                curr_model_dict[name].copy_(param)
            else:
                print("not restored",name,param.shape)
        #modelVars['model'].load_state_dict(state['state_dict'])

    # Original input size
    #if 'Dense' not in mdlParams['model_type']:
    #    print("Original input size",modelVars['model'].input_size)
    #print(modelVars['model'])

    if 'Dense' in mdlParams['model_type']:
        if mdlParams['input_size'][0] != 224:
            modelVars['model'] = modify_densenet_avg_pool(modelVars['model'])
            #print(modelVars['model'])
        num_ftrs = modelVars['model'].classifier.in_features
        modelVars['model'].classifier = nn.Linear(num_ftrs, mdlParams['numClasses'])
        #print(modelVars['model'])
    elif 'dpn' in mdlParams['model_type']:
        num_ftrs = modelVars['model'].classifier.in_channels
        modelVars['model'].classifier = nn.Conv2d(num_ftrs,mdlParams['numClasses'],[1,1])
        #modelVars['model'].add_module('real_classifier',nn.Linear(num_ftrs, mdlParams['numClasses']))
        #print(modelVars['model'])
    elif 'efficient' in mdlParams['model_type']:
        # Do nothing, output is prepared
        num_ftrs = modelVars['model']._fc.in_features
        modelVars['model']._fc = nn.Linear(num_ftrs, mdlParams['numClasses'])
    elif 'wsl' in mdlParams['model_type']:
        num_ftrs = modelVars['model'].fc.in_features
        modelVars['model'].fc = nn.Linear(num_ftrs, mdlParams['numClasses'])
    else:
        num_ftrs = modelVars['model'].last_linear.in_features
        modelVars['model'].last_linear = nn.Linear(num_ftrs, mdlParams['numClasses'])

    return modelVars

def balance_classes(mdlParams):
    if mdlParams['balance_classes'] < 3 or mdlParams['balance_classes'] == 7 or mdlParams['balance_classes'] == 11:
        # print(mdlParams['labels_array'][:4])
        # print(max(mdlParams['trainInd']))
        # class_weights = class_weight.compute_class_weight('balanced', np.unique(np.argmax(mdlParams['labels_array'][mdlParams['trainInd'], :],1)),np.argmax(mdlParams['labels_array'][mdlParams['trainInd', :]],1))
        # print("Current class weights",class_weights)
        # class_weights = class_weights*mdlParams['extra_fac']
        # print("Current class weights with extra",class_weights)
        class_weights = np.array([0.1, 0.9])
    elif mdlParams['balance_classes'] == 3 or mdlParams['balance_classes'] == 4:
        # Split training set by classes
        not_one_hot = np.argmax(mdlParams['labels_array'],1)
        mdlParams['class_indices'] = []
        for i in range(mdlParams['numClasses']):
            mdlParams['class_indices'].append(np.where(not_one_hot==i)[0])
            # Kick out non-trainind indices
            mdlParams['class_indices'][i] = np.setdiff1d(mdlParams['class_indices'][i],mdlParams['valInd'])
            #print("Class",i,mdlParams['class_indices'][i].shape,np.min(mdlParams['class_indices'][i]),np.max(mdlParams['class_indices'][i]),np.sum(mdlParams['labels_array'][np.int64(mdlParams['class_indices'][i]),:],0))
    elif mdlParams['balance_classes'] == 5 or mdlParams['balance_classes'] == 6 or mdlParams['balance_classes'] == 13:
        # Other class balancing loss
        class_weights = 1.0/np.mean(mdlParams['labels_array'][mdlParams['trainInd'],:],axis=0)
        print("Current class weights",class_weights)
        if isinstance(mdlParams['extra_fac'], float):
            class_weights = np.power(class_weights,mdlParams['extra_fac'])
        else:
            class_weights = class_weights*mdlParams['extra_fac']
        print("Current class weights with extra",class_weights)
    elif mdlParams['balance_classes'] == 9:
        # Only use official indicies for calculation
        print("Balance 9")
        indices_ham = mdlParams['trainInd'][mdlParams['trainInd'] < 25331]
        if mdlParams['numClasses'] == 9:
            class_weights_ = 1.0/np.mean(mdlParams['labels_array'][indices_ham,:8],axis=0)
            #print("class before",class_weights_)
            class_weights = np.zeros([mdlParams['numClasses']])
            class_weights[:8] = class_weights_
            class_weights[-1] = np.max(class_weights_)
        else:
            class_weights = 1.0/np.mean(mdlParams['labels_array'][indices_ham,:],axis=0)
        print("Current class weights",class_weights)
        if isinstance(mdlParams['extra_fac'], float):
            class_weights = np.power(class_weights,mdlParams['extra_fac'])
        else:
            class_weights = class_weights*mdlParams['extra_fac']
        print("Current class weights with extra",class_weights)

    return mdlParams, class_weights

