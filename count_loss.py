import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score, classification_report


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        #print("before gather",logpt)
        #print("target",target)
        logpt = logpt.gather(1,target)
        #print("after gather",logpt)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            #print("alpha",self.alpha)
            #print("gathered",at)
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def getErrClassification_mgpu(mdlParams, indices, modelVars, exclude_class=None):
    """Helper function to return the error of a set
    Args:
      mdlParams: dictionary, configuration file
      indices: string, either "trainInd", "valInd" or "testInd"
    Returns:
      loss: float, avg loss
      acc: float, accuracy
      sensitivity: float, sensitivity
      spec: float, specificity
      conf: float matrix, confusion matrix
    """
    # Set up sizes
    if indices == 'trainInd':
        numBatches = int(math.floor(len(mdlParams[indices])/mdlParams['batchSize']/len(mdlParams['numGPUs'])))
    else:
        numBatches = int(math.ceil(len(mdlParams[indices])/mdlParams['batchSize']/len(mdlParams['numGPUs'])))
    # Consider multi-crop case
    if mdlParams.get('eval_flipping',0) > 1 and mdlParams.get('multiCropEval',0) > 0:
        loss_all = np.zeros([numBatches])
        predictions = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
        targets = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
        loss_mc = np.zeros([len(mdlParams[indices])*mdlParams['eval_flipping']])
        predictions_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval'],mdlParams['eval_flipping']])
        targets_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval'],mdlParams['eval_flipping']])
        # Very suboptimal method
        ind = -1
        for i, (inputs, labels, inds, flip_ind) in enumerate(modelVars['dataloader_'+indices]):
            if flip_ind[0] != np.mean(np.array(flip_ind)):
                print("Problem with flipping",flip_ind)
            if flip_ind[0] == 0:
                ind += 1
            # Get data
            if mdlParams.get('meta_features',None) is not None:
                inputs[0] = inputs[0].cuda()
                inputs[1] = inputs[1].cuda()
            else:
                inputs = inputs.to(modelVars['device'])
            labels = labels.to(modelVars['device'])
            # Not sure if thats necessary
            modelVars['optimizer'].zero_grad()
            with torch.set_grad_enabled(False):
                # Get outputs
                if mdlParams.get('aux_classifier',False):
                    outputs, outputs_aux = modelVars['model'](inputs)
                    if mdlParams['eval_aux_classifier']:
                        outputs = outputs_aux
                else:
                    outputs = modelVars['model'](inputs)
                preds = modelVars['softmax'](outputs)
                # Loss
                loss = modelVars['criterion'](outputs, labels)
            # Write into proper arrays
            loss_mc[ind] = np.mean(loss.cpu().numpy())
            predictions_mc[ind,:,:,flip_ind[0]] = np.transpose(preds.cpu().numpy())
            tar_not_one_hot = labels.data.cpu().numpy()
            tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
            tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
            targets_mc[ind,:,:,flip_ind[0]] = np.transpose(tar)
        # Targets stay the same
        targets = targets_mc[:,:,0,0]
        # reshape preds
        print('Predictions: ', predictions)
        print('Predictions mc: ', predictions_mc)
        predictions_mc = np.reshape(predictions_mc,[predictions_mc.shape[0],predictions_mc.shape[1],mdlParams['multiCropEval']*mdlParams['eval_flipping']])
        if mdlParams['voting_scheme'] == 'vote':
            # Vote for correct prediction
            print("Pred Shape",predictions_mc.shape)
            predictions_mc = np.argmax(predictions_mc,1)
            print("Pred Shape",predictions_mc.shape)
            for j in range(predictions_mc.shape[0]):
                predictions[j,:] = np.bincount(predictions_mc[j,:],minlength=mdlParams['numClasses'])
            print("Pred Shape",predictions.shape)
        elif mdlParams['voting_scheme'] == 'average':
            predictions = np.mean(predictions_mc,2)
    elif mdlParams.get('multiCropEval',0) > 0:
        loss_all = np.zeros([numBatches])
        predictions = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
        targets = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
        loss_mc = np.zeros([len(mdlParams[indices])])
        predictions_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval']])
        targets_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval']])
        for i, (inputs, labels, inds) in enumerate(modelVars['dataloader_'+indices]):
            # Get data
            if mdlParams.get('meta_features',None) is not None:
                inputs[0] = inputs[0].cuda()
                inputs[1] = inputs[1].cuda()
            else:
                inputs = inputs.to(modelVars['device'])
            labels = labels.to(modelVars['device'])
            # Not sure if thats necessary
            modelVars['optimizer'].zero_grad()
            with torch.set_grad_enabled(False):
                # Get outputs
                if mdlParams.get('aux_classifier',False):
                    outputs, outputs_aux = modelVars['model'](inputs)
                    if mdlParams['eval_aux_classifier']:
                        outputs = outputs_aux
                else:
                    outputs = modelVars['model'](inputs)
                preds = modelVars['softmax'](outputs)
                # Loss
                loss = modelVars['criterion'](outputs, labels)
            # Write into proper arrays
            loss_mc[i] = np.mean(loss.cpu().numpy())
            predictions_mc[i,:,:] = np.transpose(preds.cpu().numpy())
            tar_not_one_hot = labels.data.cpu().numpy()
            tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
            tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
            targets_mc[i,:,:] = np.transpose(tar)
        # Targets stay the same
        targets = targets_mc[:,:,0]
        if mdlParams['voting_scheme'] == 'vote':
            # Vote for correct prediction
            print("Pred Shape",predictions_mc.shape)
            predictions_mc = np.argmax(predictions_mc,1)
            print("Pred Shape",predictions_mc.shape)
            for j in range(predictions_mc.shape[0]):
                predictions[j,:] = np.bincount(predictions_mc[j,:],minlength=mdlParams['numClasses'])
            print("Pred Shape",predictions.shape)
        elif mdlParams['voting_scheme'] == 'average':
            predictions = np.mean(predictions_mc,2)
    else:
        if mdlParams.get('model_type_cnn') is not None and mdlParams['numRandValSeq'] > 0:
            loss_all = np.zeros([numBatches])
            predictions = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
            targets = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
            loss_mc = np.zeros([len(mdlParams[indices])])
            predictions_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['numRandValSeq']])
            targets_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['numRandValSeq']])
            for i, (inputs, labels, inds) in enumerate(modelVars['dataloader_'+indices]):
                # Get data
                if mdlParams.get('meta_features',None) is not None:
                    inputs[0] = inputs[0].cuda()
                    inputs[1] = inputs[1].cuda()
                else:
                    inputs = inputs.to(modelVars['device'])
                labels = labels.to(modelVars['device'])
                # Not sure if thats necessary
                modelVars['optimizer'].zero_grad()
                with torch.set_grad_enabled(False):
                    # Get outputs
                    if mdlParams.get('aux_classifier',False):
                        outputs, outputs_aux = modelVars['model'](inputs)
                        if mdlParams['eval_aux_classifier']:
                            outputs = outputs_aux
                    else:
                        outputs = modelVars['model'](inputs)
                    preds = modelVars['softmax'](outputs)
                    # Loss
                    loss = modelVars['criterion'](outputs, labels)
                # Write into proper arrays
                loss_mc[i] = np.mean(loss.cpu().numpy())
                predictions_mc[i,:,:] = np.transpose(preds)
                tar_not_one_hot = labels.data.cpu().numpy()
                tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
                tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
                targets_mc[i,:,:] = np.transpose(tar)
            # Targets stay the same
            targets = targets_mc[:,:,0]
            if mdlParams['voting_scheme'] == 'vote':
                # Vote for correct prediction
                print("Pred Shape",predictions_mc.shape)
                predictions_mc = np.argmax(predictions_mc,1)
                print("Pred Shape",predictions_mc.shape)
                for j in range(predictions_mc.shape[0]):
                    predictions[j,:] = np.bincount(predictions_mc[j,:],minlength=mdlParams['numClasses'])
                print("Pred Shape",predictions.shape)
            elif mdlParams['voting_scheme'] == 'average':
                predictions = np.mean(predictions_mc,2)
        else:
            for i, (inputs, labels, indices) in enumerate(modelVars['dataloader_'+indices]):
                # Get data
                if mdlParams.get('meta_features',None) is not None:
                    inputs[0] = inputs[0].cuda()
                    inputs[1] = inputs[1].cuda()
                else:
                    inputs = inputs.to(modelVars['device'])
                labels = labels.to(modelVars['device'])
                # Not sure if thats necessary
                modelVars['optimizer'].zero_grad()
                with torch.set_grad_enabled(False):
                    # Get outputs
                    if mdlParams.get('aux_classifier',False):
                        outputs, outputs_aux = modelVars['model'](inputs)
                        if mdlParams['eval_aux_classifier']:
                            outputs = outputs_aux
                    else:
                        outputs = modelVars['model'](inputs)
                    #print("in",inputs.shape,"out",outputs.shape)
                    preds = modelVars['softmax'](outputs)
                    # Loss
                    loss = modelVars['criterion'](outputs, labels)
                # Write into proper arrays
                if i==0:
                    loss_all = np.array([loss.cpu().numpy()])
                    predictions = preds.cpu().numpy()
                    tar_not_one_hot = labels.data.cpu().numpy()
                    tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
                    tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
                    targets = tar
                    #print("Loss",loss_all)
                else:
                    loss_all = np.concatenate((loss_all,np.array([loss.cpu().numpy()])),0)
                    predictions = np.concatenate((predictions,preds.cpu().numpy()),0)
                    tar_not_one_hot = labels.data.cpu().numpy()
                    tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
                    tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
                    targets = np.concatenate((targets,tar),0)
                    #allInds[(i*len(mdlParams['numGPUs'])+k)*bSize:(i*len(mdlParams['numGPUs'])+k+1)*bSize] = res_tuple[3][k]
            predictions_mc = predictions
    #print("Check Inds",np.setdiff1d(allInds,mdlParams[indices]))
    # Calculate metrics
    if exclude_class is not None:
        predictions = np.concatenate((predictions[:,:exclude_class],predictions[:,exclude_class+1:]),1)
        targets = np.concatenate((targets[:,:exclude_class],targets[:,exclude_class+1:]),1)
        num_classes = mdlParams['numClasses']-1
    elif mdlParams['numClasses'] == 9 and mdlParams.get('no_c9_eval',False):
        predictions = predictions[:,:mdlParams['numClasses']-1]
        targets = targets[:,:mdlParams['numClasses']-1]
        num_classes = mdlParams['numClasses']-1
    else:
        num_classes = mdlParams['numClasses']
    # Accuarcy
    acc = np.mean(np.equal(np.argmax(predictions,1),np.argmax(targets,1)))
    print(acc)
    
    # Confusion matrix
    conf = confusion_matrix(np.argmax(targets,1),np.argmax(predictions,1))
    if conf.shape[0] < num_classes:
        conf = np.ones([num_classes,num_classes])
    # Class weighted accuracy
    wacc = conf.diagonal()/conf.sum(axis=1)
    # Sensitivity / Specificity
    sensitivity = np.zeros([num_classes])
    specificity = np.zeros([num_classes])
    # if num_classes > 2:
    #     for k in range(num_classes):
    #             sensitivity[k] = conf[k,k]/(np.sum(conf[k,:]))
    #             true_negative = np.delete(conf,[k],0)
    #             true_negative = np.delete(true_negative,[k],1)
    #             true_negative = np.sum(true_negative)
    #             false_positive = np.delete(conf,[k],0)
    #             false_positive = np.sum(false_positive[:,k])
    #             specificity[k] = true_negative/(true_negative+false_positive)
    #             # F1 score
    #             f1 = f1_score(np.argmax(predictions,1),np.argmax(targets,1),average='weighted')
    # else:
    #     tn, fp, fn, tp = confusion_matrix(np.argmax(targets,1),np.argmax(predictions,1)).ravel()
    #     sensitivity = tp/(tp+fn)
    #     specificity = tn/(tn+fp)
    #     # F1 score
    #     f1 = f1_score(np.argmax(predictions,1),np.argmax(targets,1))
    tn, fp, fn, tp = confusion_matrix(np.argmax(targets,1),np.argmax(predictions,1)).ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    acc_computed = tp+tn/(tn+fp+fn+tp)
    f1_computed = 2*tp/(2*tp +fp+fn)
    print('#######################')
    print(sensitivity,'\n', specificity, '\n', acc_computed, '\n', f1_computed)
    # F1 score
    f1 = f1_score(np.argmax(predictions,1),np.argmax(targets,1))
    # AUC
    fpr = {}
    tpr = {}
    roc_auc = np.zeros([num_classes])
    # if num_classes > 9:
    #     print(predictions)
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(targets[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return np.mean(loss_all), acc, sensitivity, specificity, conf, f1, roc_auc, wacc, predictions, targets, predictions_mc
