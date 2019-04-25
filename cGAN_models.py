import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear_z = nn.Linear(100, 200)  #   인풋 노이즈를 처리하는 linear
        self.linear_y = nn.Linear(10, 1000)  # 아웃풋 노이즈를 처리하는 linear
        self.linear = nn.Linear(1200, 28*28)

    def forward(self,z, y):
        z = self.linear_z(z)
        y = self.linear_y(y)
        cat = F.dropout(F.relu(torch.cat((z,y), dim=1)), p=0.5)
        x = self.linear(cat)
        return nn.Tanh()(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear_x = nn.Linear(784, 240)
        self.linear_y = nn.Linear(10, 50)
        self.linear_1 = nn.Linear(240+50, 240)
        self.linear_2 = nn.Linear(240,1)

    def forward(self, x, y):
        cat = F.relu(torch.cat((self.linear_x(x), self.linear_y(y)), dim=1))
        cat = F.dropout(cat, p=0.5)
        cat = F.dropout(F.relu(self.linear_1(cat)), p=0.5)
        return nn.Sigmoid()(self.linear_2(cat))

class Discriminator_Maxout(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear_x = Maxout(28 * 28, 240, k=5, dropout=False)
        self.linear_y = Maxout(10, 50, k=5, dropout=False)
        self.linear_1 = Maxout(290, 240, k=4)
        self.linear_2 = nn.Linear(240, 1)

        # "The discriminator maps x to a maxout layer with 240 units and 5 pieces, and y to a maxout layer with 50 units
        # and 5 pieces. Both of the hidden layers mapped to a joint maxout layer with 240 units and 4 pieces before
        # being fed to the sigmoid layer."

    def forward(self, x, y):
        x = torch.cat((self.linear_x(x), self.linear_y(y)), dim=1)
        x = F.dropout(self.linear_1(x), p=0.5)
        return nn.Sigmoid()(self.linear_2(x))


class Maxout(nn.Module):
    def __init__(self, in_features, out_features, k=2, dropout=True, p=0.5):
        super(Maxout, self).__init__()
        model = [nn.Dropout(p)] if dropout else []
        model += [nn.Linear(in_features, out_features * k)]

        self.model = nn.Sequential(*model)
        self.k = k

        # Note that dropout is used before weight multiplication following 'Maxout Networks' paper.
        # "When training with dropout, we perform the element-wise multiplication with the dropout mask immediately
        #  prior to the multiplication by the weights in all cases-we do not drop inputs to the max operator." - Maxout
        #  Networks

    def forward(self, x):
        x = self.model(x)
        x, _ = x.view(x.shape[0], x.shape[1] // self.k, self.k).max(-1)
        return x