import torch
from torchvision.transforms import ToTensor
import numpy as np

AA = np.array([1,2,3,4,5]).reshape(1,5)
BB = np.array([1,2,3,4,5])

AAA = ToTensor()(AA)
AAAA = torch.from_numpy(AA)
BBBB = torch.from_numpy(BB)

print(AAA)
print(AAAA)
print(BBBB)