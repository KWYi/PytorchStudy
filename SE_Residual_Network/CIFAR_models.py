import torch.nn as nn
import torch.nn.functional as F

#  Initializer
def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.detach(), mode='fan_out', nonlinearity='relu')
        # Relu 저자가 만든 initiallize 방법 kaiming_normal.

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.detach().fill_(1.)  # Batch Normalization 때의 가우시안 분산을 1로
        module.bias.detach().fill_(0.)

class View(nn.Module):
    def __init__(self, *shape):  # *의미: *가 붙으면 tuple이 됨.
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)  # *의미: 함수 안에서 쓰는 *는 tuple, list 등의 괄호를 없애줌


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, input_ch, output_ch, reduction_ratio=16, pooling='average'):
        super(SqueezeExcitationBlock, self).__init__()
        if pooling == 'average':
            pool = nn.AdaptiveAvgPool2d
        elif pooling == 'max':
            pool = nn.AdaptiveMaxPool2d
        else:
            raise NotImplementedError(print("Invalid pooling {}. Please choose among ['average', 'max']."
                                            .format(pooling)))

        block = [pool(1), View(-1), nn.Linear(input_ch, input_ch // reduction_ratio), nn.ReLU(inplace=True),
                 nn.Linear(input_ch // reduction_ratio, output_ch), nn.Sigmoid(), View(output_ch, 1, 1)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x * self.block(x)



class SEResidualBlock(nn.Module):
    def __init__(self, input_ch, output_ch, first_conv_stride=1):
        super(SEResidualBlock, self).__init__()
        block = [nn.ZeroPad2d(1), nn.Conv2d(input_ch, output_ch, 3, stride=first_conv_stride, bias=False),
                 nn.BatchNorm2d(output_ch), nn.ReLU(inplace=True)]
        block += [nn.ZeroPad2d(1), nn.Conv2d(output_ch, output_ch, 3, bias=False), nn.BatchNorm2d(output_ch),
                  SqueezeExcitationBlock(output_ch, output_ch)]
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
            network += [SEResidualBlock(16,16)]

        network += [SEResidualBlock(16, 32, first_conv_stride=2)]
        for _ in range(2):
            network += [SEResidualBlock(32,32)]

        network += [SEResidualBlock(32, 64, first_conv_stride=2)]
        for _ in range(2):
            network += [SEResidualBlock(64, 64)]

        network += [nn.AdaptiveAvgPool2d((1,1)), View(-1), nn.Linear(64, 10)]
        self.network = nn.Sequential(*network)
        self.network.apply(init_weights)  # 모든 웨이트들이 apply안에 함수를 통과함

        # 업데이트 가능한 파라미터들(grad 가짐)의 수를 모두 더해 출력
        print(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        return self.network(x)

if __name__ == '__main__':
    test = PlainNetwork()
    print(test.parameters)
