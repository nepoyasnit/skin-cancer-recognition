import torchvision.transforms as transforms


def get_train_transforms(image_size: int, train_mean: list, train_std: list):
    # create image augmentations
    transforms_train = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ]
    )

    return transforms_train


def get_valid_transforms(image_size: int, val_mean: list, val_std: list):
    val_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(val_mean, val_std)
        ]
    )

    return val_transforms


def get_base_transforms(image_size: int):
    base_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]
    )

    return base_transforms
