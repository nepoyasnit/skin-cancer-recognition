import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from augmentations import AutoAugment
from process_image import Cutout_v0

# Define ISIC Dataset Class
class ISICDataset(Dataset):
    """ISIC dataset."""

    def __init__(self, mdlParams, indSet):
        """
        Args:
            mdlParams (dict): Configuration for loading
            indSet (string): Indicates train, val, test
        """
        # Mdlparams
        self.mdlParams = mdlParams
        # Number of classes
        self.numClasses = mdlParams['numClasses']
        # Model input size
        self.input_size = (np.int32(mdlParams['input_size'][0]),np.int32(mdlParams['input_size'][1]))
        # Whether or not to use ordered cropping
        self.orderedCrop = mdlParams['orderedCrop']
        # Number of crops for multi crop eval
        self.multiCropEval = mdlParams['multiCropEval']
        # Whether during training same-sized crops should be used
        self.same_sized_crop = mdlParams['same_sized_crops']
        # Only downsample
        self.only_downsmaple = mdlParams.get('only_downsmaple',False)
        # Potential class balancing option
        self.balancing = mdlParams['balance_classes']
        # Whether data should be preloaded
        self.preload = mdlParams['preload']
        # Potentially subtract a mean
        self.subtract_set_mean = mdlParams['subtract_set_mean']
        # Potential switch for evaluation on the training set
        self.train_eval_state = mdlParams['trainSetState']
        # Potential setMean to deduce from channels
        self.setMean = mdlParams['setMean'].astype(np.float32)
        # Current indSet = 'trainInd'/'valInd'/'testInd'
        self.indices = mdlParams[indSet]
        self.indSet = indSet

        # feature scaling for meta
        self._metadata_feature_scaling()
        # Potentially preload
        if self.preload:
            self.im_list = []
            for i in range(len(self.im_paths)):
                self.im_list.append(Image.open(self.im_paths[i]))

    def _metadata_feature_scaling(self):
        if self.mdlParams.get('meta_features',None) is not None and self.mdlParams['scale_features']:
            self.feature_scaler = self.mdlParams['feature_scaler_meta']
        if self.balancing == 3 and self.indSet == 'trainInd':
            self.__train_feature_transforms()
        elif self.orderedCrop and (self.indSet == 'valInd' or self.train_eval_state  == 'eval' or self.indSet == 'testInd'):
            self.__test_crop_transforms()
        elif self.indSet == 'valInd' or self.indSet == 'testInd':
            self.__simple_test_transforms()
        else:
            self.__specific_set_transforms()

    def __train_feature_transforms(self):
            # Sample classes equally for each batch
            # First, split set by classes
            not_one_hot = np.argmax(self.mdlParams['labels_array'],1)
            self.class_indices = []
            for i in range(self.mdlParams['numClasses']):
                self.class_indices.append(np.where(not_one_hot==i)[0])
                # Kick out non-trainind indices
                self.class_indices[i] = np.setdiff1d(self.class_indices[i], self.mdlParams['valInd'])
                # And test indices
                if 'testInd' in self.mdlParams:
                    self.class_indices[i] = np.setdiff1d(self.class_indices[i], self.mdlParams['testInd'])
            # Now sample indices equally for each batch by repeating all of them to have the same amount as the max number
            indices = []
            max_num = np.max([len(x) for x in self.class_indices])
            # Go thourgh all classes
            for i in range(self.mdlParams['numClasses']):
                count = 0
                class_count = 0
                max_num_curr_class = len(self.class_indices[i])
                # Add examples until we reach the maximum
                while(count < max_num):
                    # Start at the beginning, if we are through all available examples
                    if class_count == max_num_curr_class:
                        class_count = 0
                    indices.append(self.class_indices[i][class_count])
                    count += 1
                    class_count += 1
            print("Largest class",max_num,"Indices len",len(indices))
            print("Intersect val",np.intersect1d(indices, self.mdlParams['valInd']),"Intersect Testind",np.intersect1d(indices, self.mdlParams['testInd']))
            # Set labels/inputs
            self.labels = self.mdlParams['labels_array'][indices,:]
            self.im_paths = np.array(self.mdlParams['im_paths'])[indices].tolist()
            # Normal train proc
            if self.same_sized_crop:
                cropping = transforms.RandomCrop(self.input_size)
            elif self.only_downsmaple:
                cropping = transforms.Resize(self.input_size)
            else:
                cropping = transforms.RandomResizedCrop(self.input_size[0])
            # All transforms
            self.composed = transforms.Compose([
                    cropping,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=32. / 255.,saturation=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
                    ])

    def __test_crop_transforms(self):
        # Also flip on top
        if self.mdlParams.get('eval_flipping',0) > 1:
            # Complete labels array, only for current indSet, repeat for multiordercrop
            inds_rep = np.repeat(self.mdlParams[self.indSet], self.mdlParams['multiCropEval'] * self.mdlParams['eval_flipping'])
            self.labels = self.mdlParams['labels_array'][inds_rep,:]
            # meta
            if self.mdlParams.get('meta_features',None) is not None:
                self.meta_data = self.mdlParams['meta_array'][inds_rep,:]
            # Path to images for loading, only for current indSet, repeat for multiordercrop
            self.im_paths = np.array(self.mdlParams['im_paths'])[inds_rep].tolist()
            print("len im path",len(self.im_paths))
            if self.mdlParams.get('var_im_size',False):
                self.cropPositions = np.tile(self.mdlParams['cropPositions'][self.mdlParams[self.indSet],:,:],(1, self.mdlParams['eval_flipping'],1))
                self.cropPositions = np.reshape(self.cropPositions,[self.mdlParams['multiCropEval'] * self.mdlParams['eval_flipping'] * self.mdlParams[self.indSet].shape[0],2])
                #self.cropPositions = np.repeat(self.cropPositions, (mdlParams['eval_flipping'],1))
                #print("CP examples",self.cropPositions[:50,:])
            else:
                self.cropPositions = np.tile(self.mdlParams['cropPositions'], (self.mdlParams['eval_flipping'] * self.mdlParams[self.indSet].shape[0],1))
            # Flip states
            if self.mdlParams['eval_flipping'] == 2:
                self.flipPositions = np.array([0,1])
            elif self.mdlParams['eval_flipping'] == 3:
                self.flipPositions = np.array([0,1,2])
            elif self.mdlParams['eval_flipping'] == 4:
                self.flipPositions = np.array([0,1,2,3])
            self.flipPositions = np.repeat(self.flipPositions, self.mdlParams['multiCropEval'])
            self.flipPositions = np.tile(self.flipPositions, self.mdlParams[self.indSet].shape[0])
            print("Crop positions shape",self.cropPositions.shape,"flip pos shape",self.flipPositions.shape)
            print("Flip example",self.flipPositions[:30])
        else:
            # Complete labels array, only for current indSet, repeat for multiordercrop
            inds_rep = np.repeat(self.mdlParams[self.indSet], self.mdlParams['multiCropEval'])
            self.labels = self.mdlParams['labels_array'][inds_rep,:]
            # meta
            if self.mdlParams.get('meta_features',None) is not None:
                self.meta_data = self.mdlParams['meta_array'][inds_rep,:]
            # Path to images for loading, only for current indSet, repeat for multiordercrop
            self.im_paths = np.array(self.mdlParams['im_paths'])[inds_rep].tolist()
            print("len im path",len(self.im_paths))
            # Set up crop positions for every sample
            if self.mdlParams.get('var_im_size',False):
                self.cropPositions = np.reshape(self.mdlParams['cropPositions'][self.mdlParams[self.indSet],:,:],[self.mdlParams['multiCropEval'] * self.mdlParams[self.indSet].shape[0],2])
                #print("CP examples",self.cropPositions[:50,:])
            else:
                self.cropPositions = np.tile(self.mdlParams['cropPositions'], (self.mdlParams[self.indSet].shape[0],1))
            print("CP",self.cropPositions.shape)
        #print("CP Example",self.cropPositions[0:len(mdlParams['cropPositions']),:])
        # Set up transforms
        self.norm = transforms.Normalize(np.float32(self.mdlParams['setMean']),np.float32(self.mdlParams['setStd']))
        self.trans = transforms.ToTensor()


    def __simple_test_transforms(self):
        if self.multiCropEval == 0:
            if self.only_downsmaple:
                self.cropping = transforms.Resize(self.input_size)
            else:
                self.cropping = transforms.Compose([transforms.CenterCrop(np.int32(self.input_size[0]*1.5)),transforms.Resize(self.input_size)])
            # Complete labels array, only for current indSet
            self.labels = self.mdlParams['labels_array'][self.mdlParams[self.indSet],:]
            # meta
            if self.mdlParams.get('meta_features',None) is not None:
                self.meta_data = self.mdlParams['meta_array'][self.mdlParams[self.indSet],:]
            # Path to images for loading, only for current indSet
            self.im_paths = np.array(self.mdlParams['im_paths'])[self.mdlParams[self.indSet]].tolist()
        else:
            # Deterministic processing
            if self.mdlParams.get('deterministic_eval',False):
                total_len_per_im = self.mdlParams['numCropPositions'] * len(self.mdlParams['cropScales']) * self.mdlParams['cropFlipping']
                # Actual transforms are functionally applied at forward pass
                self.cropPositions = np.zeros([total_len_per_im,3])
                ind = 0
                for i in range(self.mdlParams['numCropPositions']):
                    for j in range(len(self.mdlParams['cropScales'])):
                        for k in range(self.mdlParams['cropFlipping']):
                            self.cropPositions[ind,0] = i
                            self.cropPositions[ind,1] = self.mdlParams['cropScales'][j]
                            self.cropPositions[ind,2] = k
                            ind += 1
                # Complete labels array, only for current indSet, repeat for multiordercrop
                print("crops per image",total_len_per_im)
                self.cropPositions = np.tile(self.cropPositions, (self.mdlParams[self.indSet].shape[0],1))
                inds_rep = np.repeat(self.mdlParams[self.indSet], total_len_per_im)
                self.labels = self.mdlParams['labels_array'][inds_rep,:]
                # meta
                if self.mdlParams.get('meta_features',None) is not None:
                    self.meta_data = self.mdlParams['meta_array'][inds_rep,:]
                # Path to images for loading, only for current indSet, repeat for multiordercrop
                self.im_paths = np.array(self.mdlParams['im_paths'])[inds_rep].tolist()
            else:
                self.cropping = transforms.RandomResizedCrop(self.input_size[0],scale=(self.mdlParams.get('scale_min',0.08),1.0))
                # Complete labels array, only for current indSet, repeat for multiordercrop
                inds_rep = np.repeat(self.mdlParams[self.indSet], self.mdlParams['multiCropEval'])
                self.labels = self.mdlParams['labels_array'][inds_rep,:]
                # meta
                if self.mdlParams.get('meta_features',None) is not None:
                    self.meta_data = self.mdlParams['meta_array'][inds_rep,:]
                # Path to images for loading, only for current indSet, repeat for multiordercrop
                self.im_paths = np.array(self.mdlParams['im_paths'])[inds_rep].tolist()
        print(len(self.im_paths))
        # Set up transforms
        self.norm = transforms.Normalize(np.float32(self.mdlParams['setMean']),np.float32(self.mdlParams['setStd']))
        self.trans = transforms.ToTensor()

    def __specific_set_transforms(self):
        all_transforms = []
        # Normal train proc
        if self.same_sized_crop:
            all_transforms.append(transforms.RandomCrop(self.input_size))
        elif self.only_downsmaple:
            all_transforms.append(transforms.Resize(self.input_size))
        else:
            all_transforms.append(transforms.RandomResizedCrop(self.input_size[0],scale=(self.mdlParams.get('scale_min',0.08),1.0)))
        if self.mdlParams.get('flip_lr_ud',False):
            all_transforms.append(transforms.RandomHorizontalFlip())
            all_transforms.append(transforms.RandomVerticalFlip())
        # Full rot
        if self.mdlParams.get('full_rot',0) > 0:
            if self.mdlParams.get('scale',False):
                all_transforms.append(transforms.RandomChoice([transforms.RandomAffine(self.mdlParams['full_rot'], scale=self.mdlParams['scale'], shear=self.mdlParams.get('shear',0), interpolation=Image.NEAREST),
                                                            transforms.RandomAffine(self.mdlParams['full_rot'],scale=self.mdlParams['scale'],shear=self.mdlParams.get('shear',0), interpolation=Image.BICUBIC),
                                                            transforms.RandomAffine(self.mdlParams['full_rot'],scale=self.mdlParams['scale'],shear=self.mdlParams.get('shear',0), interpolation=Image.BILINEAR)]))
            else:
                all_transforms.append(transforms.RandomChoice([transforms.RandomRotation(self.mdlParams['full_rot'], resample=Image.NEAREST),
                                                            transforms.RandomRotation(self.mdlParams['full_rot'], resample=Image.BICUBIC),
                                                            transforms.RandomRotation(self.mdlParams['full_rot'], resample=Image.BILINEAR)]))
        # Color distortion
        if self.mdlParams.get('full_color_distort') is not None:
            all_transforms.append(transforms.ColorJitter(brightness=self.mdlParams.get('brightness_aug',32. / 255.),saturation=self.mdlParams.get('saturation_aug',0.5), contrast = self.mdlParams.get('contrast_aug',0.5), hue = self.mdlParams.get('hue_aug',0.2)))
        else:
            all_transforms.append(transforms.ColorJitter(brightness=32. / 255.,saturation=0.5))
        # Autoaugment
        if self.mdlParams.get('autoaugment',False):
            all_transforms.append(AutoAugment())
        # Cutout
        if self.mdlParams.get('cutout',0) > 0:
            all_transforms.append(Cutout_v0(n_holes=1,length=self.mdlParams['cutout']))
        # Normalize
        all_transforms.append(transforms.ToTensor())
        all_transforms.append(transforms.Normalize(np.float32(self.mdlParams['setMean']),np.float32(self.mdlParams['setStd'])))
        # All transforms
        self.composed = transforms.Compose(all_transforms)
        # Complete labels array, only for current indSet
        self.labels = self.mdlParams['labels_array'][self.mdlParams[self.indSet],:]
        # meta
        if self.mdlParams.get('meta_features',None) is not None:
            self.meta_data = self.mdlParams['meta_array'][self.mdlParams[self.indSet],:]
        # Path to images for loading, only for current indSet
        self.im_paths = np.array(self.mdlParams['im_paths'])[self.mdlParams[self.indSet]].tolist()


    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # Load image
        if self.preload:
            x = self.im_list[idx]
        else:
            x = Image.open(self.im_paths[idx])
            if self.mdlParams.get('resize_large_ones',0) > 0 and (x.size[0] == self.mdlParams['large_size'] and x.size[1] == self.mdlParams['large_size']):
                width = self.mdlParams['resize_large_ones']
                height = self.mdlParams['resize_large_ones']
                #height = (self.mdlParams['resize_large_ones']/self.mdlParams['large_size'])*x.size[1]
                x = x.resize((width,height),Image.BILINEAR)
            if self.mdlParams['input_size'][0] >= 224 and self.mdlParams['orderedCrop']:
                if x.size[0] < self.mdlParams['input_size'][0]:
                    new_height = int(self.mdlParams['input_size'][0]/float(x.size[0]))*x.size[1]
                    new_width = self.mdlParams['input_size'][0]
                    x = x.resize((new_width,new_height),Image.BILINEAR)
                if x.size[1] < self.mdlParams['input_size'][0]:
                    new_width = int(self.mdlParams['input_size'][0]/float(x.size[1]))*x.size[0]
                    new_height = self.mdlParams['input_size'][0]
                    x = x.resize((new_width,new_height),Image.BILINEAR)
        # Get label
        y = self.labels[idx,:]
        # meta
        if self.mdlParams.get('meta_features',None) is not None:
            x_meta = self.meta_data[idx,:].copy()
        # Transform data based on whether train or not train. If train, also check if its train train or train inference
        if self.orderedCrop and (self.indSet == 'valInd' or self.indSet == 'testInd' or self.train_eval_state == 'eval'):
            # Apply ordered cropping to validation or test set
            # Get current crop position
            x_loc = self.cropPositions[idx,0]
            y_loc = self.cropPositions[idx,1]
            # scale
            if self.mdlParams.get('meta_features',None) is not None and self.mdlParams['scale_features']:
                x_meta = np.squeeze(self.feature_scaler.transform(np.expand_dims(x_meta,0)))
            if self.mdlParams.get('trans_norm_first',False):
                # First, to pytorch tensor (0.0-1.0)
                x = self.trans(x)
                # Normalize
                x = self.norm(x)
                #print(self.im_paths[idx])
                #print("Before",x.size(),"xloc",x_loc,"y_loc",y_loc)
                if self.mdlParams.get('eval_flipping',0) > 1:
                    if self.flipPositions[idx] == 1:
                        x = torch.flip(x,(1,))
                    elif self.flipPositions[idx] == 2:
                        x = torch.flip(x,(2,))
                    elif self.flipPositions[idx] == 3:
                        x = torch.flip(x,(1,2))
                #print((x_loc-np.int32(self.input_size[0]/2.)),(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],(y_loc-np.int32(self.input_size[1]/2.)),(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1])
                x = x[:,np.int32(x_loc-(self.input_size[0]/2.)):np.int32(x_loc-(self.input_size[0]/2.))+self.input_size[0],
                        np.int32(y_loc-(self.input_size[1]/2.)):np.int32(y_loc-(self.input_size[1]/2.))+self.input_size[1]]
                #print("After",x.size())
            else:
                # Then, apply current crop
                #print("Before",x.size(),"xloc",x_loc,"y_loc",y_loc)
                #print((x_loc-np.int32(self.input_size[0]/2.)),(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],(y_loc-np.int32(self.input_size[1]/2.)),(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1])
                x = Image.fromarray(np.array(x)[(x_loc-np.int32(self.input_size[0]/2.)):(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],
                        (y_loc-np.int32(self.input_size[1]/2.)):(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1],:])
                # First, to pytorch tensor (0.0-1.0)
                x = self.trans(x)
                # Normalize
                x = self.norm(x)
            #print("After",x.size())
        elif self.indSet == 'valInd' or self.indSet == 'testInd':
            if self.mdlParams.get('deterministic_eval',False):
                crop = self.cropPositions[idx,0]
                scale = self.cropPositions[idx,1]
                flipping = self.cropPositions[idx,2]
                if flipping == 1:
                    # Left flip
                    x = transforms.functional.hflip(x)
                elif flipping == 2:
                    # Right flip
                    x = transforms.functional.vflip(x)
                elif flipping == 3:
                    # Both flip
                    x = transforms.functional.hflip(x)
                    x = transforms.functional.vflip(x)
                # Scale
                if int(scale*x.size[0]) > self.input_size[0] and int(scale*x.size[1]) > self.input_size[1]:
                    x = transforms.functional.resize(x,(int(scale*x.size[0]),int(scale*x.size[1])))
                else:
                    x = transforms.functional.resize(x,(self.input_size[0],self.input_size[1]))
                # Crop
                if crop == 0:
                    # Center
                    x = transforms.functional.center_crop(x,self.input_size[0])
                elif crop == 1:
                    # upper left
                    x = transforms.functional.crop(x, self.mdlParams['offset_crop']*x.size[0], self.mdlParams['offset_crop']*x.size[1], self.input_size[0],self.input_size[1])
                elif crop == 2:
                    # lower left
                    x = transforms.functional.crop(x, self.mdlParams['offset_crop']*x.size[0], (1.0-self.mdlParams['offset_crop'])*x.size[1]-self.input_size[1], self.input_size[0],self.input_size[1])
                elif crop == 3:
                    # upper right
                    x = transforms.functional.crop(x, (1.0-self.mdlParams['offset_crop'])*x.size[0]-self.input_size[0], self.mdlParams['offset_crop']*x.size[1], self.input_size[0],self.input_size[1])
                elif crop == 4:
                    # lower right
                    x = transforms.functional.crop(x, (1.0-self.mdlParams['offset_crop'])*x.size[0]-self.input_size[0], (1.0-self.mdlParams['offset_crop'])*x.size[1]-self.input_size[1], self.input_size[0],self.input_size[1])
            else:
                x = self.cropping(x)
            # To pytorch tensor (0.0-1.0)
            x = self.trans(x)
            x = self.norm(x)
            # scale
            if self.mdlParams.get('meta_features',None) is not None and self.mdlParams['scale_features']:
                x_meta = np.squeeze(self.feature_scaler.transform(np.expand_dims(x_meta,0)))
        else:
            # Apply
            x = self.composed(x)
            # meta augment
            if self.mdlParams.get('meta_features',None) is not None:
                if self.mdlParams['drop_augment'] > 0:
                    # randomly deactivate a feature
                    # age
                    if torch.rand(1) < self.mdlParams['drop_augment']:
                        if 'age_oh' in self.mdlParams['meta_features']:
                            x_meta[0:self.mdlParams['meta_feature_sizes'][0]] = np.zeros([self.mdlParams['meta_feature_sizes'][0]])
                        else:
                            x_meta[0] = -5
                    if torch.rand(1) < self.mdlParams['drop_augment']:
                        if 'loc_oh' in self.mdlParams['meta_features']:
                            x_meta[self.mdlParams['meta_feature_sizes'][0]:self.mdlParams['meta_feature_sizes'][0]+self.mdlParams['meta_feature_sizes'][1]] = np.zeros([self.mdlParams['meta_feature_sizes'][1]])
                    if torch.rand(1) < self.mdlParams['drop_augment']:
                        if 'sex_oh' in self.mdlParams['meta_features']:
                            x_meta[self.mdlParams['meta_feature_sizes'][0]+self.mdlParams['meta_feature_sizes'][1]:self.mdlParams['meta_feature_sizes'][0]+self.mdlParams['meta_feature_sizes'][1]+self.mdlParams['meta_feature_sizes'][2]] = np.zeros([self.mdlParams['meta_feature_sizes'][2]])
                # scale
                if self.mdlParams['scale_features']:
                    x_meta = np.squeeze(self.feature_scaler.transform(np.expand_dims(x_meta,0)))
        # Transform y
        y = np.argmax(y)
        y = np.int64(y)
        if self.mdlParams.get('meta_features',None) is not None:
            x_meta = np.float32(x_meta)
        if self.mdlParams.get('eval_flipping',0) > 1:
            return x, y, idx, self.flipPositions[idx]
        else:
            if self.mdlParams.get('meta_features',None) is not None:
                return (x, x_meta), y, idx
            else:
                return x, y, idx