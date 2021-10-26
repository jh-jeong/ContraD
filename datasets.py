import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms

DATA_PATH = os.environ.get('DATA_DIR', 'data/')

class IndexedDataset(Dataset):
    """
    Wraps another dataset to sample from. Returns the sampled indices during iteration.
    In other words, instead of producing (X, y) it produces (X, y, idx)
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return (img, label, idx)

def get_dataset(dataset, indexed=False):
    if dataset == 'cifar10' or dataset == 'cifar100':
        image_size = (32, 32, 3)
        transform = transforms.ToTensor()

        if dataset == 'cifar10':
            data = datasets.CIFAR10
        else:
            data = datasets.CIFAR100

        train_set = data(DATA_PATH, train=True, transform=transform, download=True)
        test_set = data(DATA_PATH, train=False, transform=transform, download=True)
        if indexed:
            train_set = IndexedDataset(train_set)
            test_set = IndexedDataset(test_set)
        return train_set, test_set, image_size

    elif dataset == 'cifar10_lin' or dataset == 'cifar100_lin':
        """CIFAR-10/100 for linear evaluation.
        We follow the augmentation scheme used in [1] specially for linear evaluation. 
        
        [1] https://github.com/HobbitLong/SupContrast
        """
        if indexed:
            raise ValueError("Have not implemented indexed dataset")

        image_size = (32, 32, 3)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

        if dataset == 'cifar10_lin':
            data = datasets.CIFAR10
        else:
            data = datasets.CIFAR100

        train_set = data(DATA_PATH, train=True, transform=train_transform, download=True)
        test_set = data(DATA_PATH, train=False, transform=test_transform, download=True)

        return train_set, test_set, image_size

    elif dataset == 'cifar10_hflip' or dataset == 'cifar100_hflip':
        """CIFAR-10/100 with HFlip augmentation.
        Only used for training DiffAug models as per [1].
        
        [1] Zhao et al., Differentiable Augmentation for Data-efficient GAN Training, NeurIPS 2020.
        """
        if indexed:
            raise ValueError("Have not implemented indexed dataset")

        image_size = (32, 32, 3)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        if dataset == 'cifar10_hflip':
            data = datasets.CIFAR10
        else:
            data = datasets.CIFAR100
        train_set = data(DATA_PATH, train=True, transform=train_transform, download=True)
        test_set = data(DATA_PATH, train=False, transform=transforms.ToTensor(), download=True)

        return train_set, test_set, image_size

    elif dataset == 'celeba128':
        if indexed:
            raise ValueError("Have not implemented indexed dataset")

        image_size = (128, 128, 3)
        data_path = f"{DATA_PATH}/CelebAMask-HQ/CelebA-128-split"

        train_dir = os.path.join(data_path, 'train')
        test_dir = os.path.join(data_path, 'test')

        train_set = datasets.ImageFolder(train_dir, transforms.ToTensor())
        test_set = datasets.ImageFolder(test_dir, transforms.ToTensor())

        return train_set, test_set, image_size

    elif dataset == 'afhq_cat':
        if indexed:
            raise ValueError("Have not implemented indexed dataset")

        image_size = (512, 512, 3)

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        train_dir = os.path.join(DATA_PATH, 'afhq/cat/train')
        val_dir = os.path.join(DATA_PATH, 'afhq/cat/val')

        train_set = datasets.ImageFolder(train_dir, train_transform)
        val_set = datasets.ImageFolder(val_dir, transforms.ToTensor())

        return train_set, val_set, image_size

    elif dataset == 'afhq_dog':
        if indexed:
            raise ValueError("Have not implemented indexed dataset")

        image_size = (512, 512, 3)

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        train_dir = os.path.join(DATA_PATH, 'afhq/dog/train')
        val_dir = os.path.join(DATA_PATH, 'afhq/dog/val')

        train_set = datasets.ImageFolder(train_dir, train_transform)
        val_set = datasets.ImageFolder(val_dir, transforms.ToTensor())

        return train_set, val_set, image_size

    elif dataset == 'afhq_wild':
        if indexed:
            raise ValueError("Have not implemented indexed dataset")

        image_size = (512, 512, 3)

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        train_dir = os.path.join(DATA_PATH, 'afhq/wild/train')
        val_dir = os.path.join(DATA_PATH, 'afhq/wild/val')

        train_set = datasets.ImageFolder(train_dir, train_transform)
        val_set = datasets.ImageFolder(val_dir, transforms.ToTensor())

        return train_set, val_set, image_size


def get_dataset_ref(dataset):
    if dataset == 'cifar10' or dataset == 'cifar100':
        if dataset == 'cifar10':
            data = datasets.CIFAR10
        else:
            data = datasets.CIFAR100
        reference = data(DATA_PATH, train=False, transform=transforms.ToTensor(), download=True)

    elif dataset == 'cifar10_hflip' or dataset == 'cifar100_hflip':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        if dataset == 'cifar10_hflip':
            data = datasets.CIFAR10
        else:
            data = datasets.CIFAR100
        reference = data(DATA_PATH, train=False, transform=transform, download=True)

    elif dataset == 'celeba128':
        data_path = f"{DATA_PATH}/CelebAMask-HQ/CelebA-128-split/test"
        reference = datasets.ImageFolder(data_path, transforms.ToTensor())

    elif dataset == 'afhq_cat':
        data_path = f'{DATA_PATH}/afhq/cat/train'
        reference = datasets.ImageFolder(data_path, transforms.ToTensor())
    elif dataset == 'afhq_dog':
        data_path = f'{DATA_PATH}/afhq/dog/train'
        reference = datasets.ImageFolder(data_path, transforms.ToTensor())
    elif dataset == 'afhq_wild':
        data_path = f'{DATA_PATH}/afhq/wild/train'
        reference = datasets.ImageFolder(data_path, transforms.ToTensor())
    else:
        raise NotImplementedError()

    return reference