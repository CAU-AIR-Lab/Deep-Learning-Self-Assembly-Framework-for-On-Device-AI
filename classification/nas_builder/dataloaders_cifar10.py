import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]


def get_loaders(train_portion, path_to_save_data, batch_size, img_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(img_size, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_data = datasets.CIFAR10(root=path_to_save_data, train=True,
                                  download=True, transform=train_transform)
    num_train = len(train_data)  
    indices = np.arange(num_train)
    np.random.shuffle(indices)
    split = int(np.floor(train_portion * num_train))

    train_idx, valid_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler, num_workers=2)

    if train_portion == 1:
        return train_loader

    valid_sampler = SubsetRandomSampler(valid_idx)

    val_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=2)

    return train_loader, val_loader


def get_test_loader(path_to_save_data, batch_size, img_size):
    test_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    test_data = datasets.CIFAR10(root=path_to_save_data, train=False,
                                 download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False, num_workers=16)
    return test_loader
