import os
import numpy as np
import PIL
import random

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Lambda, ToTensor
from PIL import Image

class CustomCIFAR10(Dataset):
    def __init__(self, train=True):
        super(CustomCIFAR10, self).__init__()
        self.cifar10_train = CIFAR10(root='datasets', train = train, download=True)
        #  self._cifar10_train => index[0] = data, index[1] = label, format = PIL

        images = list()
        for i in range(len(self.cifar10_train)):
            images.append(np.array(self.cifar10_train[i][0]))
        self.per_pixel_mean_grid = np.mean(images, axis=0).astype(np.float32)
        print(self.per_pixel_mean_grid.shape)

        if not train:
            self.cifar10_test = CIFAR10(root='datasets', train=train, download=False)

        self.train = train

    def __getitem__(self, index):
        transforms = list()
        transforms.append(Lambda(self.__to_numpy))
        transforms.append(Lambda(self.__per_pixel_mean_normalization))

        if self.train:
            # if random.random() > 0.5:
            #     transforms.append(Lambda(self.__horizontal_flip))
            transforms.append(Lambda(self.__pad_and_random_crop))
        transforms.append(ToTensor())
        transforms = Compose(transforms)

        if self.train:
            return transforms(self.cifar10_train[index][0]), self.cifar10_train[index][1]
        else:
            return transforms(self.cifar10_test[index][0]), self.cifar10_test[index][1]

    def __len__(self):
        if self.train:
            return len(self.cifar10_train)
        else:
            return len(self.cifar10_test)

    # Static Methods
    def __to_numpy(self, x):
        assert isinstance(x, PIL.Image.Image)  # assert: 뒤의 객체가 True가 아니면 Error Raise
        return np.array(x).astype(np.float32)

    def __per_pixel_mean_normalization(self, x):
        return (x-self.per_pixel_mean_grid)/255.

    def __pad_and_random_crop(self, x):
        p = 4
        x = np.pad(x, ((p,p), (p,p), (0,0)), mode='constant', constant_values=0.)
        y_index = random.randint(0, 2*p -1)
        x_index = random.randint(0, 2*p -1)
        x = x[y_index: y_index+32, x_index: x_index+32, :]
        return x

if __name__=='__main__':
    dataset = CustomCIFAR10()