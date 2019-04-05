## In here, we load pre-maden model from train_20190316_1.py and catagorize testset using that model

import torch
from torch.utils.data import DataLoader
import torchvision
from train_20190316_1 import MLP
#train_20190316_1 모델의 MLP 클래스를 불러옴.
# 일반적인 경우, train_20190316_1 파일의 구문이 전부 실행된 후 MLP를 가져온다. 이는 우리가 원하는 것이 아니기 때문에
# train_20190316_1 안에 if '__name__' == __main__' 구문 추가

# Input pipeline
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),  #데이터셋의 정규화를 train과 똑같게 설정
                                            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])
dataset = torchvision.datasets.MNIST(root='datasets', train=False, transform=transform, download=True)  # MNIST의 테스트셋 불러옴
data_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, num_workers=0)  # DataLoader 를 통해 데이터를 읽어옴

# Define model
mlp = MLP()  # MLP의 인스턴스 mlp 생성. 이는 MLP 클래스의 설정 모델 구조를 불러옴.
trained_model = torch.load('mlp.pt')  # train_20190316_1 에서 저장한 가중치 파일 mlp.pt를 불러옴
state_dict = trained_model.state_dict()  # 훈련된 모델을 불러온 trained_model 의 상태(weight, bias)를 state_dict로 저장
                                         # dictionary 데이터이므로 state_dict.keys(), .values(), .items() 등으로 읽을 수 있음
mlp.load_state_dict(state_dict)  # MLP 클래스의 모델 구조를 가진 mlp 객체에 state_dict의 가중치&바이어스를 입력함

nb_correct_answers = 0  # 정답을 맞춘 갯수를 저장하기 위한 객체
for data in data_loader:  # data 객체로 data_loader의 성분을 불러옴
    input, label = data[0], data[1]  # data의 인풋과 라벨을 불러옴
    input = input.view(input.shape[0], -1)  # 인풋 데이터를 1차원 행렬로 만듬
            # [batch size, channel*height*weight]
    classification_results = mlp(input)  # mlp.pt의 가중치를 입력받은 MLP 클래스의 인스턴스 mlp 에 인풋을 입력
    nb_correct_answers += torch.eq(classification_results.argmax(), label).sum()  # torch.eq(x,y): x,y가 같으면 1 출력
    # .sum을 통해 한 batch 내에서 맞춘 갯수를 출력. .sum() 없으면 batch_size의 행렬에 각 성분마다 정답이면1, 정답이 아니면 0 돌려줌.([0,0,1,1,0,0,1])
print("Average acc.: {} %.".format(float(nb_correct_answers) / len(data_loader) *100))  # 정답률 출력