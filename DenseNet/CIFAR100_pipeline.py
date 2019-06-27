import os
import numpy as np
import PIL
import random

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, Lambda, ToTensor, RandomHorizontalFlip, Pad, RandomCrop, Normalize
from PIL import Image

class CustomCIFAR100(Dataset):
    def __init__(self, train=True):
        super(CustomCIFAR100, self).__init__()
        self.cifar_100 = CIFAR100(root='datasets', train = train, download=True)
        #  self._cifar10_train => index[0] = data, index[1] = label, format = PIL

        tensors = list()
        for i in range(len(self.cifar_100)):
            tensors.append(ToTensor()(self.cifar_100[i][0]).numpy())  # cifar100
            # ToTensor 는 들어온 이미지나 ndarray(H, W, C)를 (C, H, W)의 텐서로 바꿔서 돌려줌
        mean = np.mean(tensors, axis=(0,2,3))  # 채널 별 평균 구하기. ToTensor를 썼기 때문에 index=1이 채널
        std = np.std(tensors, axis=(0,2,3))  # 채널 별 표준편차 구하기. ToTensor를 썼기 때문에 index=1이 채널
        print("mean: {}, std: {}".format(mean, std))

        transform = [RandomHorizontalFlip()]
        transform += [Pad(4), RandomCrop(32)]
        transform += [ToTensor(), Normalize(mean=mean, std=std)]
        self.transform = Compose(transform)

    def __getitem__(self, index):  # __getitem__ : index로 value를 얻을 수 있게 하는 super method
        tensor, label = self.transform(self.cifar_100[index][0]), self.cifar_100[index][1]
        return tensor, label

    def __len__(self):  # __getitem__ : len()로 데이터의 길이를 얻을 수 있게 하는 super method
        return len(self.cifar_100)

if __name__=='__main__':
    import os
    dataset = CustomCIFAR100()
    print(dataset[0])