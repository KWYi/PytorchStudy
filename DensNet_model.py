import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, n_ch, growth_rate):
        # growth_rate: DenseNet은 concat으로 인해 각 블럭의 output이 더해지면서 Dense_Block의 input이 점점 커짐.
        # 각 block의 output 사이즈이자, 인풋이 점점 늘어나는 크기가 growth_rate
        super(DenseLayer, self).__init__()
        layer = []
        layer += [nn.BatchNorm2d(n_ch),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(n_ch, 4*growth_rate, kernel_size=1, bias=False)]
                # 1X1 Convolution의 output은 4*growth_rate로 고정 => 채널 감소 효과
        layer += [nn.BatchNorm2d(4*growth_rate),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)]
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        y = self.layer(x)
        return torch.cat((y,x), dim=1)

class DenseBlock(nn.Module):
    def __init__(self, n_layers, n_ch, growth_rate):
        super(DenseBlock, self).__init__()
        # A way.
        # self.Dense_layer_0 = DenseLayer(n_ch, growth_rate)
        # self.Dense_layer_1 = DenseLayer(n_ch+growth_rate, growth_rate)
        # self.Dense_layer_2 = DenseLayer(n_ch+2*growth_rate, growth_rate)
        for i in range(n_layers):
            # B way.
            setattr(self, 'Dense_layer_{}'.format(i), DenseLayer(n_ch+i*growth_rate, growth_rate))

            # C way.
            # self.add_module('Dense_layer_{}'.format((i),DenseLayer(n_ch+i*growth_rate, growth_rate)))

        # A, B, C ways are same.
        self.n_layers = n_layers

    def forward(self, x):
        for i in range(self.n_layers):
            x = getattr(self, 'Dense_layer_{}'.format(i))(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, n_ch):
        super(TransitionLayer, self).__init__()
        layer = [nn.BatchNorm2d(n_ch),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(n_ch, n_ch//2, kernel_size=1, bias=False),
                 nn.AvgPool2d(kernel_size=2, stride=2)]

        self.layer = nn.Sequential(*layer)

    def forward(self,x):
        return self.layer(x)

class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, input_ch=3, n_classes=100):
        super(DenseNet, self).__init__()
        n_ch = 2*growth_rate
        n_layers = 16
        network = [nn.Conv2d(input_ch, n_ch, kernel_size=3, padding=1, bias=False)]
        network += [DenseBlock(16, n_ch, growth_rate)]
        n_ch = n_ch + n_layers * growth_rate

        network += [TransitionLayer(n_ch)]
        network += [DenseBlock(16, n_ch//2, growth_rate)]
        n_ch = n_ch//2 + n_layers * growth_rate

        network += [TransitionLayer(n_ch)]
        network += [DenseBlock(16, n_ch//2, growth_rate)]
        n_ch = n_ch//2 + n_layers * growth_rate

        network += [nn.BatchNorm2d(n_ch),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),  # Batch_size * n_ch * 1*1  4차원 행렬
                    View(-1),  # Batch_size * data 로 flattening
                    nn.Linear(n_ch, n_classes)]

        self.network = nn.Sequential(*network)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.weight, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.network(x)

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

if __name__ == '__main__':
    from CIFAR100_pipeline import CustomCIFAR100
    from torch.utils.data import DataLoader

    dataset = CustomCIFAR100()
    dataloader = DataLoader(dataset, batch_size=1)
    model = DenseNet()

    for tensor, label in dataloader:
        model(tensor)
        break
