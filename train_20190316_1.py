import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import trange



class MLP(nn.Module):  # MLP 클래스 생성
    def __init__(self):  # 클래스 생성 시 실행할 작업들 선언
        super(MLP, self).__init__()
        # super 목적1. 조상 클래스 중 두 개 이상의 클래스를 상속받은 클래스가 있을 시,
        #   그 두 클래스의 내용을 모두 상속받기 위하여 선언
        # super 목적2. 후에 이 클래스를 포함하여 두 개 이상의 클래스를 상속받는 자식 클래스가 있을 경우,
        #   그 자식 클래스에 이 클래스의 특징을 온전히 전달하기 위하여 선언
        self.fc1 = nn.Linear(in_features=28*28, out_features=64)  # 28*28 이미지를 인풋으로 받아 64개의 아웃풋을 돌려주는 가중치 행렬(786*60) 만들고 아웃풋 출력
        self.fc2 = nn.Linear(in_features=64, out_features=128)  # 1*60 행렬을 인풋으로 받아 128개의 아웃풋을 돌려주는 가중치 행렬(60*128) 만들고 아웃풋 출력
        self.fc3 = nn.Linear(in_features=128, out_features=256)  # 1*128 행렬을 인풋으로 받아 256개의 아웃풋을 돌려주는 가중치 행렬(128*256)  만들고 아웃풋 출력
        self.fc4 = nn.Linear(in_features=256, out_features=10)  # 1*256 행렬을 인풋으로 받아 10개(구분하려는 범주의 갯수)의
                                                                 # 아웃풋을 돌려주는 가중치 행렬(60*10) 만들고 아웃풋 출력

    def forward(self, x):  # forward 부함수 선언
        x = nn.ReLU()(self.fc1(x))  # __init__ 에서 만든 fc1 가중치 행렬을 이용하여 x에 가중치 계산한 후 ReLU 활성화 함수 적용
        x = F.relu(self.fc2(x))  # __init__ 에서 만든 fc2 가중치 행렬을 이용하여 x에 가중치 계산한 후 ReLU 활성화 함수 적용
        # F.relu 랑 nn.relu() 랑 차이 없음
        x = nn.ReLU()(self.fc3(x))  # __init__ 에서 만든 fc3 가중치 행렬을 이용하여 x에 가중치 계산한 후 ReLU 활성화 함수 적용
        x = self.fc4(x)  # __init__ 에서 만든 fc4 가중치 행렬을 이용하여 x에 가중치 계산한 후 출력. 마지막 출력값이니 활성화 함수 불필요.
        return x  # x 되돌려줌


if __name__=='__main__':  # 이하의 구문이 이 파일내에서 직접 실행시킬 때만 작동하도록 하는 if문


    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])
    # transform. 이용할 데이터를 어떻게 조절할지 선언. ToTensor(): 주어진 데이터를 Tensor 형식으로 만듬.
    #   Normalize(mean=[0.5], std[0.5]): 정규표준편차 (x-mean)/std 에서, x는 0~1 범위의 정규화 된 데이터, mean=0이나
    #   mean = 0.5로 정의하여 x-mean 의 범위를 -0.5~0.5 로 만들고, 이를 0.5로 나누어 -1~+1 범위로 표준편차 정규화
    dataset = torchvision.datasets.MNIST(root='datasets', train=True, transform = transform, download=True)
    # 'dataset' 폴더의 MNIST 데이터를 호출. root:경로, train=학습 웨이트 함께 가져옴, transform=위에서 정의한 transform으로 정규화
    # download=True : 지정 경로에 다운로드. download= False 일 시 다운로드 안받고 존재하는 MNIST 데이터 사용.


    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, num_workers=0, shuffle=True)
    # 반복 작업을 위한 준비를 하는 문구. dataset = 데이터, batch_size = 한 번에 훈련시킬 데이터수,
    #  num_workers = 데이터를 읽는데 사용할 cpu thread 갯수, shuffle=데이터 순서를 뒤섞을지 여부

    mlp = MLP()  # mlp 라는 인스턴스로 MLP 클래스 호출
    loss = nn.CrossEntropyLoss()  # loss 객체로 CrossEntropyLoss 함수 선언
    # CrossEntropy 실행 시 먼저 자동으로 softmax 를 실행함.
    optim = torch.optim.Adam(mlp.parameters(), lr=2e-4, betas=[0.5, 0.99])
    # 옵티마이저 선언. / lr:러닝 레이트. 웨이트가 변할 때 gradient를 얼마나 반영할 지 정함.
    # beta1: std / beta2: Adam 공식의 분모에 들어갈 log(x) 의 x가 0이 되는 것을 막기 위해 부여하는 bias 수치. 맞나? 확인할 것.


    EPOCHS = 6  # 데이터 전체를 몇 번이나 이용해 학습할 지 저장하기 위한 객체
    total_step = 0  # 총 몇 번의 학습이 일어났나 저장하기 위한 객체
    list_loss = list()  # loss 저장용 리스트

    for epoch in trange(EPOCHS):  # EPOCHS 만큼 실행할 for 문
        for i, data in enumerate(data_loader):  # data_loader 에서 선언한 data 와 그 인덱스 i를 선언
            total_step += 1  # step을 1 늘림
            input, label = data[0], data[1]  # data_loader에서 읽은 data 객체의 인풋 데이터와 라벨 읽어옴
            # input shape = [32,1,28,28]  [batch size, channel, height, width]
            input = input.view(input.shape[0], -1)  # 인풋 데이터를 1차원 행렬로 만듬
            # [batch size, channel*height*weight]

            classification_results = mlp.forward(input)  # [batch size, 10]
            #  MLP 클래스의 forward 부함수들 실행. 입력 데이터는 위에서 만든 input

            l = loss(classification_results, label)  # loss 객체(Cross entropy) 에 forward 결과값과 라벨 입력
            list_loss.append(l.detach().item())  # loss_list 에 loss 저장.
            # l.detach(): 로스 계산을 하지 않겠다는 선언. 이미 l = loss 에서 로스를 계산했으니 필요 없음.
            # item(): tensor 형식으로 돌려줄 값을 float 형식으로 돌려줌

            optim.zero_grad()  # make gradients zero. 각 웨이트가 부여받은 gradient를 0으로 만듬
            l.backward()  # giving weight's gradient to each weights.  l 객체에서 구한 loss 값을 가중치들에 부여
            optim.step()  # Weights adjust themself using given gradient.
            # 각 객체가 부여받은 loss 값을 선언한 옵티마이저 설정에 맞게 수정하여 가중치에 반영

    torch.save(mlp, 'mlp.pt')  # mlp.pt 라는 이름으로 모델의 가중치&바이어스 파일 저장

    plt.figure()
    plt.plot(range(len(list_loss)), list_loss)
    plt.show()
