import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):  # MLP 클래스 생성
    def __init__(self):  # 클래스 생성 시 실행할 작업들 선언
        super(MLP, self).__init__()
        # super 목적1. 조상 클래스 중 두 개 이상의 클래스를 상속받은 클래스가 있을 시,
        #   그 두 클래스의 내용을 모두 상속받기 위하여 선언
        # super 목적2. 후에 이 클래스를 포함하여 두 개 이상의 클래스를 상속받는 자식 클래스가 있을 경우,
        #   그 자식 클래스에 이 클래스의 특징을 온전히 전달하기 위하여 선언
        self.fc1 = nn.Linear(in_features=28 * 28,
                             out_features=64)  # 28*28 이미지를 인풋으로 받아 64개의 아웃풋을 돌려주는 가중치 행렬(786*60) 만들고 아웃풋 출력
        self.fc2 = nn.Linear(in_features=64,
                             out_features=128)  # 1*60 행렬을 인풋으로 받아 128개의 아웃풋을 돌려주는 가중치 행렬(60*128) 만들고 아웃풋 출력
        self.fc3 = nn.Linear(in_features=128,
                             out_features=256)  # 1*128 행렬을 인풋으로 받아 256개의 아웃풋을 돌려주는 가중치 행렬(128*256)  만들고 아웃풋 출력
        self.fc4 = nn.Linear(in_features=256, out_features=10)  # 1*256 행렬을 인풋으로 받아 10개(구분하려는 범주의 갯수)의
        # 아웃풋을 돌려주는 가중치 행렬(60*10) 만들고 아웃풋 출력

    def forward(self, x):  # forward 부함수 선언
        x = nn.ReLU()(self.fc1(x))  # __init__ 에서 만든 fc1 가중치 행렬을 이용하여 x에 가중치 계산한 후 ReLU 활성화 함수 적용
        x = F.relu(self.fc2(x))  # __init__ 에서 만든 fc2 가중치 행렬을 이용하여 x에 가중치 계산한 후 ReLU 활성화 함수 적용
        # F.relu 랑 nn.relu() 랑 차이 없음
        x = nn.ReLU()(self.fc3(x))  # __init__ 에서 만든 fc3 가중치 행렬을 이용하여 x에 가중치 계산한 후 ReLU 활성화 함수 적용
        x = self.fc4(x)  # __init__ 에서 만든 fc4 가중치 행렬을 이용하여 x에 가중치 계산한 후 출력. 마지막 출력값이니 활성화 함수 불필요.
        return x  # x 되돌려줌


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input shape : [Batch_size, 1, 28, 28]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=2)
        # output shape : [Batch_size, 16, 14, 14]
        # round(28-3+2*1)/s + 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2)
        # output shape : [Batch_size, 32, 7, 7]
        # round(14-3+2*1)/s + 1
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)
        # output shape : [Batch_size, 64, 4, 4]
        # round(7-3+2*1)/s + 1
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        # output shape : [Batch_size, 128, 2, 2]
        # round(4-3+2*1)/s + 1
        self.linear = nn.Linear(128 * 2 * 2, 10)  # nn.linear(input, output)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x