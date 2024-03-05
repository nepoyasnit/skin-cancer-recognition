import numpy as np
import torch
from PIL import Image


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self,
        image_paths: list,
        targets: np.array,
        resize: int,
        augmentations=None,
        backend="pil",
        channel_first=True):
        """
        :param image_paths: list of paths to images
        :param targets: numpy array
        :param resize: tuple or None
        :param augmentations: albumentations augmentations
        """
        super().__init__()
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.backend = backend

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        targets = self.targets[item]
        if self.backend == "pil":
            image = Image.open(self.image_paths[item]).convert("RGB")
            image = np.float32(image)
            if self.augmentations is not None:
                image = self.augmentations(image=image)["image"].astype(np.float32)
            else:
                image = image.astype(np.float32)
            image = image.transpose(2, 0, 1)
        else:
            raise Exception("Backend not implemented")
        return image, targets
