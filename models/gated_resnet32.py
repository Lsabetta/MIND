'''
Reference:
https://github.com/khurramjaved96/incremental-learning/blob/autoencoders/model/resnet32.py
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from copy import deepcopy
from parse import args as OPT

class GatedConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mask = torch.ones(self.weight.shape)

    def forward(self, input: Tensor) -> Tensor:
        self.mask = self.mask.to(self.weight.device)
        return self._conv_forward(input, self.weight*self.mask, self.bias)


class GatedLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mask = torch.ones(self.weight.shape).to(self.weight.device)

    def forward(self, input: Tensor) -> Tensor:
        self.mask = self.mask.to(self.weight.device)
        return F.linear(input, self.weight*self.mask, self.bias)
        #return F.linear(input, self.weight*self.mask, torch.zeros_like(self.bias).to(self.weight.device))
    

    
class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.conv = GatedConv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut, track_running_stats=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DownsampleC(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = GatedConv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsampleD(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        assert stride == 2
        self.conv = GatedConv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut, track_running_stats=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,dropout_rate=0.1):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = GatedConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes, track_running_stats=True)

        self.conv_b = GatedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes, track_running_stats=True)

        self.downsample = downsample

        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.dropout(basicblock)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.dropout(F.relu(residual + basicblock, inplace=True))


class GatedCifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, channels=3,dropout_rate=0.1,classes_out=100):
        super(GatedCifarResNet, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.conv_1_3x3 = GatedConv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16, track_running_stats=True)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1,dropout_rate=dropout_rate)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2,dropout_rate=dropout_rate)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2,dropout_rate=dropout_rate)
        #self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.avgpool = nn.AvgPool2d(8)
        if 'CORE50' in OPT.dataset:
            expansion = 4 # 4*4
        elif OPT.dataset == 'TinyImageNet':
            expansion = 4
        else:
            expansion = block.expansion

        #expansion = block.expansion

        self.out_dim = 64 * block.expansion
        self.fc = GatedLinear(64*expansion, classes_out, bias=False)
        self.output_mask = {}
        self.exp_idx = -1
        self.bn_weights = {}
        self.dropout = torch.nn.Dropout(p= dropout_rate)


        for m in self.modules():
            if isinstance(m, GatedConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, GatedLinear):
                nn.init.kaiming_normal_(m.weight)
                #m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,dropout_rate=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,dropout_rate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dropout_rate=dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x = self.dropout(F.relu(self.bn_1(x), inplace=True))

        x_1 = self.stage_1(x)  # [bs, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        x_3 = self.stage_3(x_2)  # [bs, 64, 8, 8]

        pooled = self.avgpool(x_3)  # [bs, 64, 1, 1]
        features = pooled.view(pooled.size(0), -1)  # [bs, 64]

        # return {
        #     'fmaps': [x_1, x_2, x_3],
        #     'features': features
        # }
        return self.fc(features) * self.output_mask[self.exp_idx]

        
    def forward_fts(self, x):
        x = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x = F.relu(self.bn_1(x), inplace=True)

        x_1 = self.stage_1(x)  # [bs, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        x_3 = self.stage_3(x_2)  # [bs, 64, 8, 8]

        pooled = self.avgpool(x_3)  # [bs, 64, 1, 1]
        features = pooled.view(pooled.size(0), -1)  # [bs, 64]

        # return {
        #     'fmaps': [x_1, x_2, x_3],
        #     'features': features
        # }
        return features
        
    @property
    def last_conv(self):
        return self.stage_3[-1].conv_b



    def set_output_mask(self, exp_idx, classes_in_this_exp):
        # set zero to all the output neurons that are not in this experience
        self.exp_idx = exp_idx
        self.output_mask[exp_idx] = torch.nn.functional.one_hot( torch.tensor(classes_in_this_exp), num_classes=OPT.n_classes).sum(dim=0).float().to(OPT.device)           

    def save_bn_params(self, task_id):
        """Save the BN weights of the model in a dict"""
        bn_params = []
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                bn_params.append((m.weight.data.clone(), m.bias.data.clone(), m.running_mean.clone(), m.running_var.clone()))
        self.bn_weights[task_id] = bn_params


    def load_bn_params(self, task_id):
        """Load the BN weights saved in the dict"""
        bn_paramz = deepcopy(self.bn_weights[task_id])
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data, m.bias.data, m.running_mean, m.running_var = bn_paramz.pop(0)


def gresnet20mnist():
    """Constructs a ResNet-20 model for MNIST."""
    model = GatedCifarResNet(ResNetBasicblock, 20, 1)
    return model


def gresnet32mnist():
    """Constructs a ResNet-32 model for MNIST."""
    model = GatedCifarResNet(ResNetBasicblock, 32, 1)
    return model


def gresnet20():
    """Constructs a ResNet-20 model for CIFAR-10."""
    model = GatedCifarResNet(ResNetBasicblock, 20)
    return model


def gresnet32(dropout_rate=0.1,classes_out=OPT.n_classes):
    """Constructs a ResNet-32 model for CIFAR-10."""
    model = GatedCifarResNet(ResNetBasicblock, 32,dropout_rate=dropout_rate,classes_out=classes_out)
    return model


def gresnet44():
    """Constructs a ResNet-44 model for CIFAR-10."""
    model = GatedCifarResNet(ResNetBasicblock, 44)
    return model


def gresnet56():
    """Constructs a ResNet-56 model for CIFAR-10."""
    model = GatedCifarResNet(ResNetBasicblock, 56)
    return model


def gresnet110():
    """Constructs a ResNet-110 model for CIFAR-10."""
    model = GatedCifarResNet(ResNetBasicblock, 110)
    return model

# for auc
def gresnet14():
    model = GatedCifarResNet(ResNetBasicblock, 14)
    return model

def gresnet26():
    model = GatedCifarResNet(ResNetBasicblock, 26)
    return model






