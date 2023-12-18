import numpy as np
import torch
from PIL import Image

class Cutout_v0(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        img = np.array(img)
        #print(img.shape)
        h = img.shape[0]
        w = img.shape[1]

        mask = np.ones((h, w), np.uint8)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        #mask = torch.from_numpy(mask)
        #mask = mask.expand_as(img)
        img = img * np.expand_dims(mask,axis=2)
        img = Image.fromarray(img)
        return img

# Sampler for balanced sampling
class StratifiedSampler(torch.utils.data.sampler.Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, mdlParams):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.dataset_len = len(mdlParams['trainInd'])
        self.numClasses = mdlParams['numClasses']
        self.trainInd = mdlParams['trainInd']
        # Sample classes equally for each batch
        # First, split set by classes
        not_one_hot = np.argmax(mdlParams['labels_array'][mdlParams['trainInd'],:],1)
        self.class_indices = []
        for i in range(mdlParams['numClasses']):
            self.class_indices.append(np.where(not_one_hot==i)[0])
        self.current_class_ind = 0
        self.current_in_class_ind = np.zeros([mdlParams['numClasses']],dtype=int)

    def gen_sample_array(self):
        # Shuffle all classes first
        for i in range(self.numClasses):
            np.random.shuffle(self.class_indices[i])
        # Construct indset
        indices = np.zeros([self.dataset_len],dtype=np.int32)
        ind = 0
        while(ind < self.dataset_len):
            indices[ind] = self.class_indices[self.current_class_ind][self.current_in_class_ind[self.current_class_ind]]
            # Take care of in-class index
            if self.current_in_class_ind[self.current_class_ind] == len(self.class_indices[self.current_class_ind])-1:
                self.current_in_class_ind[self.current_class_ind] = 0
                # Shuffle
                np.random.shuffle(self.class_indices[self.current_class_ind])
            else:
                self.current_in_class_ind[self.current_class_ind] += 1
            # Take care of overall class ind
            if self.current_class_ind == self.numClasses-1:
                self.current_class_ind = 0
            else:
                self.current_class_ind += 1
            ind += 1
        return indices

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return self.dataset_len