import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def tranCon3x3(in_planes, out_planes, output_padding=0, stride=1):
    print(stride)
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=3,
                              stride=stride,
                              padding=1,
                              output_padding=output_padding,
                              bias=False)


class BasicEncodeBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicEncodeBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicDecodeBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicDecodeBlock, self).__init__()
        if stride > 1:
            output_padding = 1
        else:
            output_padding = 0

        self.conv1 = tranCon3x3(inplanes, planes, output_padding, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = tranCon3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetAutoencoder(nn.Module):
    def __init__(self, encode_block, decode_block, layers):
        super(ResNetAutoencoder, self).__init__()
        self.inplanes = 1

        self.encode_layer1 = self._make_encode_layer(encode_block, 1, layers[0])
        self.encode_layer2 = self._make_encode_layer(encode_block, 64, layers[1], stride=2)
        self.encode_layer3 = self._make_encode_layer(encode_block, 128, layers[2], stride=2)
        self.encode_layer4 = self._make_encode_layer(encode_block, 256, layers[3], stride=2)

        self.decode_layer1 = self._make_decode_layer(decode_block, 256, layers[3], stride=2)
        self.decode_layer2 = self._make_decode_layer(decode_block, 128, layers[2], stride=2)
        self.decode_layer3 = self._make_decode_layer(decode_block, 64, layers[1], stride=2)
        self.decode_layer4 = self._make_decode_layer(decode_block, 1, layers[0])

    def _make_encode_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_decode_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride > 1:
            output_padding = 1
        else:
            output_padding = 0

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block.expansion, output_padding=output_padding, kernel_size=1, stride=stride, bias=False),
            )
        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)

        # encoding
        x = self.encode_layer1(x)
        # print(x.shape)
        x = self.encode_layer2(x)
        # print(x.shape)
        x = self.encode_layer3(x)
        # print(x.shape)
        x = self.encode_layer4(x)
        # print(x.shape)

        # decoding
        x = self.decode_layer1(x)
        # print(x.shape)
        x = self.decode_layer2(x)
        # print(x.shape)
        x = self.decode_layer3(x)
        # print(x.shape)
        x = self.decode_layer4(x)
        # print(x.shape)

        # x = x.view(x.size(0), -1)

        return x