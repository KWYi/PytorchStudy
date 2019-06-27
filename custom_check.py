if __name__ == '__main__':
    import os
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision.transforms import Compose, ToTensor, Normalize
    from torchvision.utils import save_image
    import datetime

    BATCH_SIZE = 16
    EPOCHS = 10
    IMAGE_DIR = 'C:\\Users\\KangwooYi\\PycharmProjects\\PytorchLearning\\GANs\\checkpoints\\MNIST\\Images\\Training'
    IMAGE_SIZE = 28
    ITER_DISPLAY = 100
    ITER_REPORT = 100
    LATENT_DIM = 100
    MODEL_DIR = 'C:\\Users\\KangwooYi\\PycharmProjects\\PytorchLearning\\GANs\\checkpoints\\MNIST\\Models'
    OUT_CHANNEL = 1
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    transforms = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    dataset = MNIST(root='datasets', train=True, transform=transforms, download=True)

    print(dataset.__dict__)