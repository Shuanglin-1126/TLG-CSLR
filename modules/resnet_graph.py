import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size=5, window_stride=1, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size - 1) * (window_dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x):
        # Input shape: (N,C,T,H,W), out: (N,C,T,V*window_size)
        N, C, T, H, W = x.shape
        x = x.flatten(-2)
        x = self.unfold(x)  # (N, C*Window_Size, T, P)
        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        x = x.view(-1, C, self.window_size, T, H*W).permute(0, 3, 1, 2, 4).reshape(N*T, C, -1)  # (NT)C(SP)
        return x


class Correlation_Module(nn.Module):
    def __init__(self, k=5, nighs=7):
        super().__init__()
        self.k = k
        self.nighs = nighs
        self.init_decay = -0.1

    def forward(self, x, upfold):
        L, N, D = x.shape
        affinities = torch.einsum('bdl,bdn->bln', x, upfold) / math.sqrt(D)

        _, indices = torch.topk(affinities, self.k*self.nighs, dim=1)  # (L, k, N, D)
        mask = torch.zeros_like(affinities, dtype=torch.float32)
        mask.scatter_(1, indices, 1.)

        # affinities = torch.softmax(affinities, dim=-1) * mask
        affinities = torch.sigmoid(affinities) * mask / (self.k * self.nighs)  # 非 top-k 的地方自动乘成 0
        features = torch.einsum('bln,bdn->bdl', affinities, upfold)

        return features


class TemporalAggregationBlock(nn.Module):
    def __init__(self, d_model: int, nighs: int = 7, k: int = 9, stride: int = 1):
        super().__init__()

        self.attn = Correlation_Module(k)
        padding = (kernel_size - 1) // 2
        self.conv_in = nn.Conv2d(d_model, d_model, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(d_model)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(d_model, d_model, kernel_size=1, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(d_model)
        self.d_model = d_model
        self.upfold = UnfoldTemporalWindows(nighs)
        self.weights = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.apply(self.init_weights_xavier)

    def init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            # 使用 nn.init.xavier_uniform_ 初始化权重
            nn.init.xavier_uniform_(m.weight)
            # 偏置通常初始化为 0
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def attention(self, x: torch.Tensor):

        x_upfold = self.upfold(x)  # LND -> NDL -> LND
        return self.attn(x.flatten(-2), x_upfold)

    def forward(self, x: torch.Tensor, T):
        NT, C, H, W = x.shape
        residual = x

        x = self.conv_in(x)
        x = self.relu(self.bn1(x))
        x = x.view(-1, T, C, H, W).transpose(1, 2)
        x = self.attention(x).view(NT, C, H, W)
        x = self.conv2(x)
        x = self.bn2(x)

        x += residual * self.weights
        x = self.relu(x)

        return x


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.tggbloks = TemporalAggregationBlock
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # bt, c

        return x


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    model.load_state_dict(checkpoint, strict=False)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet50'])
    model.load_state_dict(checkpoint, strict=False)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


def test():
    net = resnet18()
    y = net(torch.randn(1, 3, 224, 224))
    print(y.size())

# test()