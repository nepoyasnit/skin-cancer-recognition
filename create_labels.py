import numpy as np
from glob import glob
import csv

def set_labels(mdlParams, labels_str):
    '''
    Set labels by .csv file
    '''
    for label_row in labels_str:
        if 'image_name' == label_row[0]:
            continue
        #if 'ISIC' in row[0] and '_downsampled' in row[0]:
        #    print(row[0])
        if label_row[0] + '_downsampled' in mdlParams['labels_dict']:
            print("removed", label_row[0] + '_downsampled')
            continue
        if mdlParams['numClasses'] == 1:
            if label_row[1] == 1:
                mdlParams['labels_dict'][label_row[0]] = np.array([1])
            else:
                mdlParams['labels_dict'][label_row[0]] = np.array([0])
        if mdlParams['numClasses'] == 2:
            if label_row[1] == '0':
                mdlParams['labels_dict'][label_row[0]] = np.array([1, 0])
            else:
                mdlParams['labels_dict'][label_row[0]] = np.array([0, 1])
            # if label_row[1] == '1':
            #     mdlParams['labels_dict'][label_row[0]] = np.array([1])
            # else:
            #     mdlParams['labels_dict'][label_row[0]] = np.array([0])
        elif mdlParams['numClasses'] == 7:
            mdlParams['labels_dict'][label_row[0]] = np.array([int(float(label_row[i])) for i in range(1,8)])
        elif mdlParams['numClasses'] == 8:
            if len(label_row) < 9 or label_row[8] == '':
                class_8 = 0
            else:
                class_8 = int(float(label_row[8]))
            mdlParams['labels_dict'][label_row[0]] = np.array([int(float(label_row[i])) for i in range(1,9)])
        elif mdlParams['numClasses'] == 9:
            mdlParams['labels_dict'][label_row[0]] = np.array([int(float(label_row[i])) for i in range(1,10)])
    return mdlParams

def create_labels_dict(mdlParams):
    '''
    Create dict with labels like {'ISIC_0000000': array([0, 1, 0, 0, 0, 0, 0, 0, 0]), ...}
    '''
    path1 = mdlParams['dataDir'] + '/labels/'
    print('Labels path: ', path1)
        # All sets
    allSets = glob(path1 + '*/')
    print(allSets)
    # Go through all sets
    for i in range(len(allSets)):
        # Check if want to include this dataset
        foundSet = False
        for j in range(len(mdlParams['dataset_names'])):
            if mdlParams['dataset_names'][j] in allSets[i]:
                foundSet = True
        if not foundSet:
            continue
        # Find csv file
        files = sorted(glob(allSets[i]+'*'))
        for j in range(len(files)):
            if 'csv' in files[j]:
                break
        # Load csv file
        with open(files[j], newline='') as csvfile:
            labels_str = csv.reader(csvfile, delimiter=',', quotechar='|')
            mdlParams = set_labels(mdlParams, labels_str)

    return mdlParams
