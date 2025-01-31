import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils import data

from utils import *



transforms_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(
        (32, 32),
        padding=4,
        fill=0,
        padding_mode="constant"
    ),

    transforms.RandomHorizontalFlip(),

    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
    ),
    FlattenAndCast(),
])


transforms_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        ),
        FlattenAndCast(),
    ]
)

train_dataset = CIFAR10(
    root="./CIFAR", train=True, download=True, transform=transforms_train
)

test_dataset = CIFAR10(
    root="./CIFAR", train=False, download=True, transform=transforms_test
)



def get_dataloaders(
    batch_size=128,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
):

    return NumpyLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    ), NumpyLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )