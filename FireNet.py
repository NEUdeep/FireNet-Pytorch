# author: johnkang
# time: 05.15.2020/05.20 clock

from __future__ import division
import torch
import torch.nn as nn
import math


class FireNet(nn.Module):
    def __init__(self, features, num_classes=2):
        """
        args:
        features: cnns main construct using cfg to make
        num_classes: numbers of classes

        """
        super(FireNet, self).__init__()
        self.features = features
        self.fc = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
             nn.Softmax()
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  # torch.Size([2, 64, 8, 8])

        x = x.view(x.size(0), -1)  # torch.Size([2, 4096])

        x = self.fc(x)
        # print(x.size())
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = {
    'F14': [16, 'A', 32, 'A', 64, 'A'],
}


def make_layers(cfg, bn=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if bn:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)




def FireNet14():
    model = FireNet(make_layers(cfg['F14']), num_classes=2)
    return model


if __name__ == "__main__":
    data = torch.autograd.Variable(torch.randn(2, 3, 64, 64))
    print(data)
    net = FireNet14()
    print(net)
    output = net.forward(data)
    print(output)  # tensor([[ 0.0009, -0.0008]], grad_fn=<AddmmBackward>)
    # tensor([[0.4999, 0.5001]], grad_fn=<SoftmaxBackward>) softmax


