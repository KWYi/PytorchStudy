import torch.nn as nn
import torch.functional as F
from torch.functional import F

#  Initializer
def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.detach(), mode='fan_out', nonlinearity='relu')
        # Relu 저자가 만든 initiallize 방법 kaiming_normal.

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.detach().fill_(1.)  # Batch Normalization 때의 가우시안 분산을 1로
        module.bias.detach().fill_(0.)


class PlainNetwork(nn.Module):
    def __init__(self):
        super(PlainNetwork, self).__init__()
        act = nn.ReLU(inplace = True)
        pad = nn.ZeroPad2d
        norm = nn.BatchNorm2d

        network = []
        network += [pad(1), nn.Conv2d(3, 16, 3, bias=False), norm(16), act]
        # nn.Conv2d(input channel, output_channel, kernal_size)
        # bias=False 이유: BatchNorm 을 쓸건데, BatchNorm이 Bias 추가를 함
        for _ in range(6):
            network += [pad(1), nn.Conv2d(16, 16, 3, bias=False), norm(16), act]

        network += [pad(1), nn.Conv2d(16, 32, 3, stride=2, bias=False), norm(32), act]
        for _ in range(5):
            network += [pad(1), nn.Conv2d(32, 32, 3, bias=False), norm(32), act]

        network += [pad(1), nn.Conv2d(32, 64, 3, stride=2, bias=False), norm(64), act]
        for _ in range(5):
            network += [pad(1), nn.Conv2d(64, 64, 3, stride=2, bias=False), norm(64), act]

        network += [nn.AdaptiveAvgPool2d((1,1)), View(-1), nn.Linear(64, 10)]
        # AdaptiveAvgPool2d : 64(channel)*8*8 결과를 채널별로 평균을 냄. => 채널별 8*8의 평균인 64*1*1 행렬이 됨.
        # 이걸 1층 Dense를 통하여 10개의 아웃풋을 뽑아냄. (CIFAR10 데이터의 라벨이 10개)
        # 이렇게 하는게 Dense를 여러게 쌓은 것보다 성능이 비슷하거나 더 좋으면서 계산량과 오버피팅 위험을 피할 수 있다.

        self.netwrok = nn.Sequential(*network)
        # Sequential은 forward가 정의된 Instance만을 통과시킬 수 있음
        # 그래서 우리는 아래에 View class를 만들고 그 안에 forward 부함수를 만들어 줄거임

        self.apply(init_weights)  # 모든 웨이트들이 apply안에 함수를 통과함

        # 업데이트 가능한 파라미터들(grad 가짐)의 수를 모두 더해 출력

        print(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        return self,network(x)

class View(nn.Module):
    def __init__(self, *shape):  # *의미: *가 붙으면 tuple이 됨.
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)  # *의미: 함수 안에서 쓰는 *는 tuple, list 등의 괄호를 없애줌

class ResidualBlock(nn.Module):
    def __init__(self, input_ch, output_ch, first_conv_stride=1):
        super(ResidualBlock, self).__init__()
        block = [nn.ZeroPad2d(1), nn.Conv2d(input_ch, output_ch, 3, stride=first_conv_stride, bias=False),
                 nn.BatchNorm2d(output_ch), nn.ReLU(inplace=True)]
        block += [nn.ZeroPad2d(1), nn.Conv2d(output_ch, output_ch, 3, bias=False), nn.BatchNorm2d(output_ch)]
        self.block = nn.Sequential(*block)

        if first_conv_stride >1:
            self.varying_size = True
            side_block = [nn.ZeroPad2d(1), nn.MaxPool2d(kernel_size=3, stride=2),
                         nn.ConstantPad3d((0,0,0,0,0, output_ch - input_ch), value=0.)]
                       # nn.ConstantPad3d((위,아래,좌,우,앞,뒤), values=0.). 3d 어레이 어디에 0값을 추가할지 선언
            self.side_block = nn.Sequential(*side_block)

        else:
            self.varying_size = False

    def forward(self, x):
        if self.varying_size:
            return F.relu(self.side_block(x) + self.block(x))

        else:
            return F.relu(x + self.block(x))

class ResidualNetwork(nn.Module):
    def __init__(self):
        super(ResidualNetwork, self).__init__()
        network = [nn.ZeroPad2d(1), nn.Conv2d(3, 16, 3, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True)]
        for _ in range(3):
            network += [ResidualBlock(16,16)]

        network += [ResidualBlock(16, 32, first_conv_stride=2)]
        for _ in range(2):
            network += [ResidualBlock(32,32)]

        network += [ResidualBlock(32, 64, first_conv_stride=2)]
        for _ in range(2):
            network += [ResidualBlock(64, 64)]

        network += [nn.AdaptiveAvgPool2d((1,1)), View(-1), nn.Linear(64, 10)]
        self.network = nn.Sequential(*network)
        self.apply(init_weights)  # 모든 웨이트들이 apply안에 함수를 통과함

        # 업데이트 가능한 파라미터들(grad 가짐)의 수를 모두 더해 출력
        print(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        return self.network(x)

if __name__ == '__main__':
    test = PlainNetwork()
    print(test.parameters)