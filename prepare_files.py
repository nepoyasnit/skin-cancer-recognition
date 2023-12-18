from glob import glob
import numpy as np
import scipy

def make_official_dataset_first(allSets, mdlParams):
    '''
    Sets the processing of the official data set first
    '''
    for i in range(len(allSets)):
        if mdlParams['dataset_names'][0] in allSets[i]:
            temp = allSets[i]
            allSets.remove(allSets[i])
            allSets.insert(0, temp)

    print('SETS: ', allSets)
    return allSets

def get_preload_imgs(mdlParams):
    '''
    Set preload imgs
    '''
    mdlParams['images_array'] = np.zeros([len(mdlParams['im_paths']),mdlParams['input_size_load'][0],mdlParams['input_size_load'][1],mdlParams['input_size_load'][2]],dtype=np.uint8)
    for i in range(len(mdlParams['im_paths'])):
        x = scipy.ndimage.imread(mdlParams['im_paths'][i])
        #x = x.astype(np.float32)
        # Scale to 0-1
        #min_x = np.min(x)
        #max_x = np.max(x)
        #x = (x-min_x)/(max_x-min_x)
        mdlParams['images_array'][i,:,:,:] = x
        if i%1000 == 0:
            print(i+1,"images loaded...")
    return mdlParams


def set_images_dirs(allSets, mdlParams, exclude_list, indices_exclude):
    '''
    Set images dirs and labels in params
    '''
    for i in range(len(allSets)):
    # All files in that set
        files = sorted(glob(allSets[i]+'*'))
        # Check if there is something in there, if not, discard
        if len(files) == 0:
            continue
        # Check if want to include this dataset
        foundSet = False
        for j in range(len(mdlParams['dataset_names'])):
            if mdlParams['dataset_names'][j] in allSets[i]:
                foundSet = True
        if not foundSet:
            continue
        print(len(files))
        for j in range(len(files)):
            if '.jpg' in files[j] or '.jpeg' in files[j] or '.JPG' in files[j] or '.JPEG' in files[j] or '.png' in files[j] or '.PNG' in files[j]:
                # Add according label, find it first
                found_already = False
                for key in mdlParams['labels_dict']:
                    if key + mdlParams['file_ending'] in files[j]:
                        if found_already:
                            print("Found already:",key,files[j])
                        mdlParams['key_list'].append(key)
                        mdlParams['labels_list'].append(mdlParams['labels_dict'][key])
                        found_already = True
                if found_already:
                    mdlParams['im_paths'].append(files[j])
                    if mdlParams['exclude_inds']:
                        for key in indices_exclude:
                            if key in files[j]:
                                exclude_list.append(indices_exclude[key])

    return allSets, mdlParams, exclude_list

