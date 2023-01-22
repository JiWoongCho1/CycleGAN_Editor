import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True,
                 **kwargs):  # **kwargs : 정해진 키워드 파라미터 받을 수 있음.
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)
            if down else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()  # ReLU를 쓰지 않으면 그냥 그대로 흘려보낸다.
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_resblock=9):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            # 같은 크기, 채널의깊이만 달라진다.
            nn.ReLU(inplace=True)  # 처음에 노이즈를 넣는 것이 아닌 이미지를 넣는다.
        )
        self.down_blocks = nn.ModuleList(
            [  # --> 절반으로 크기를 줄인다.
                ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1), ]
        )
        self.residual_block = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_resblock)]  # 9개의 잔차블럭.
        )

        self.up_blocks = nn.ModuleList(
            [  # --> 이미지크기 2배로 늘림림
                ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1),
                ConvBlock(num_features * 2, num_features, down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1)
            ])

        self.last = nn.Conv2d(num_features * 1, img_channels, kernel_size=7, stride=1, padding=3,
                              padding_mode='reflect')


    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_block(x)
        for layer in self.up_blocks:
            x = layer(x)

        return torch.tanh(self.last(x))