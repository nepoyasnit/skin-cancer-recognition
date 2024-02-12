import torchvision.transforms as transforms
from .constants import IMG_SIZE


def get_transforms(image_size: int, rgb_mean: list, rgb_std: list):
    # create image augmentations
    transforms_train = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std),
        ]
    )

    transforms_valid = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std),
        ]
    )

    return transforms_train, transforms_valid
