import torch
import random
import numpy as np
import torchvision
import lightning as L
from torchvision import transforms
from typing import Optional, List, Tuple
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Sampler

from models import configure_random_seed


def seed_worker(worker_id):
    random_seed = torch.initial_seed() % 2**32
    configure_random_seed(random_seed)


class CustomSampler(Sampler):
    def __init__(self, data_source, seeds):
        self.data_source = data_source
        self.seeds = seeds

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seeds)
        indices = torch.randperm(len(self.data_source), generator=g).tolist()
        return iter(indices)

    def __len__(self):
        return len(self.data_source)


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        data_augmentation: bool = True,
        resize: int = 0,
        random_seed: Optional[int] = None,
    ):
        """
        Initializes the CIFAR10DataModule for loading and processing the CIFAR-10 dataset.

        Parameters:
        data_dir (str): The directory where the CIFAR-10 dataset is stored or will be downloaded.
        batch_size (int): The size of each data batch.
        data_augmentation (bool): Whether to apply data augmentation to the training data.
        resize (int): If greater than zero, resize the images to the specified size.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.resize = resize
        self.random_seed = random_seed

        configure_random_seed(random_seed)

        common_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        if resize > 0:
            common_transforms.insert(0, transforms.Resize(resize))

        if data_augmentation:
            self.transform_train = transforms.Compose(
                [
                    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                    transforms.RandomHorizontalFlip(),
                    *common_transforms,
                ]
            )
        else:
            self.transform_train = transforms.Compose(common_transforms)

        self.transform_test = transforms.Compose(common_transforms)

        self.cifar10_train = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform_train,
        )
        self.cifar10_val = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform_test,
        )
        self.cifar10_test = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform_test,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar10_train,
            batch_size=self.batch_size,
            num_workers=2,
            persistent_workers=True,
            shuffle=True if self.random_seed is None else None,
            sampler=(
                CustomSampler(self.cifar10_train, self.random_seed)
                if self.random_seed is not None
                else None
            ),
            worker_init_fn=seed_worker,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar10_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            worker_init_fn=seed_worker,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar10_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            worker_init_fn=seed_worker,
        )


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 512,
        random_seed: Optional[int] = None,
    ):
        """
        Initializes the MNISTDataModule for loading and processing the MNIST dataset.

        Parameters:
        data_dir (str): The directory where the MNIST dataset is stored or will be downloaded.
        batch_size (int): The size of each data batch.
        data_augmentation (bool): Whether to apply data augmentation to the training data.
        resize (int): If greater than zero, resize the images to the specified size.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.random_seed = random_seed

        configure_random_seed(random_seed)

        common_transforms = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        self.transform_train = transforms.Compose(common_transforms)
        self.transform_test = transforms.Compose(common_transforms)

        self.mnist_train = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform_train,
        )
        self.mnist_val = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform_test,
        )
        self.mnist_test = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform_test,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=2,
            persistent_workers=True,
            shuffle=True if self.random_seed is None else None,
            sampler=(
                CustomSampler(self.mnist_train, self.random_seed)
                if self.random_seed is not None
                else None
            ),
            worker_init_fn=seed_worker,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            worker_init_fn=seed_worker,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            worker_init_fn=seed_worker,
        )
