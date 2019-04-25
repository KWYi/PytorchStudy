import os
import random
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Grayscale, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root, crop_size=0, flip=False):
        super(CustomDataset, self).__init__()
        self.root = root
        self.list_path = os.listdir(root)
        self.crop_size = crop_size
        self.flip = flip

    def __getitem__(self, index):  # __getitem__ 매직메소드가 있어야만 인덱싱 가능.
        image = Image.open(os.path.join(self.root+self.list_path[index]))
        list_transforms = list()
        if self.crop_size >0:
            list_transforms.append(RandomCrop((self.crop_size, self.crop_size)))
        if self.flip:
            coin = random.random() > 0.5
            if coin:
                list_transforms.append(RandomHorizontalFlip())

        transforms = Compose(list_transforms)

        image = transforms(image)
        input_image = Grayscale(num_output_channels=1)(image)
        input_tensor = ToTensor()(input_image)
        target_tensor = ToTensor()(image)

        input_tensor = Normalize(mean=[0.5], std=[0.5])(input_tensor)
        target_tensor = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(target_tensor)

        return input_tensor, target_tensor

    def __len__(self):
        return len(self.list_path)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import numpy as np
    dir_image = 'dataset\\images\\'
    dataset = CustomDataset(root=dir_image, crop_size = 128, flip=True)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=2)

    for input, target in data_loader:
        input_image_np = np.array(input[0,0])
        # input은 batch*channel*height*width 차원이나, 우리는 여기서 인풋 이미지를 눈으로 보고자 함으로 batch와 channel 차원은 없어애 한다
        # input[0,0] 으로 설정함으로서 1번째 batch, 1번째 channel 의 height, width 데이터만 불러올 수 있다.
        input_image_np -= input_image_np.min()
        input_image_np /= input_image_np.max()
        input_image_np *= 255.0
        input_image_np = input_image_np.astype(np.uint8)
        input_image = Image.fromarray(input_image_np)
        input_image.show()

        target_image_np = np.array(target[0])
        target_image_np -= target_image_np.min()
        target_image_np /= target_image_np.max()
        target_image_np *= 255.0
        target_image_np = target_image_np.astype(np.uint8)
        target_image = Image.fromarray(target_image_np.transpose(1,2,0), mode='RGB')
        target_image.show()

        break