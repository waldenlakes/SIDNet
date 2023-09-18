import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import nn

from model.blocks import RDB

class IlluminationModule(nn.Module):

    def __init__(self, channels_in=3, channels_out=96):
        super(IlluminationModule, self).__init__()

        self.nChannel_in = channels_in
        self.nChannel_out = channels_out
        self.nDenselayer = 8
        self.nFeat = 32
        scale = 2
        self.growthRate = 16

        # F-1
        self.conv1 = nn.Conv2d(self.nChannel_in, self.nFeat, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(self.nFeat, self.nFeat, kernel_size=3, padding=1, bias=True)

        self.down1 = nn.MaxPool2d(2)  # 384 - 192

        # dense layers
        self.dense1_1 = RDB(self.nFeat, self.nDenselayer, self.growthRate)
        self.dense1_2 = RDB(self.nFeat, self.nDenselayer, self.growthRate)
        self.d_ch1 = nn.Conv2d(self.nFeat, self.nFeat * 2, kernel_size=1, padding=0, bias=True)

        self.down2 = nn.MaxPool2d(2)  # 192 - 96

        self.dense2_1 = RDB(self.nFeat * 2, self.nDenselayer, self.growthRate)
        self.dense2_2 = RDB(self.nFeat * 2, self.nDenselayer, self.growthRate)
        self.d_ch2 = nn.Conv2d(self.nFeat * 2, self.nFeat * 4, kernel_size=1, padding=0, bias=True)

        self.down3 = nn.MaxPool2d(2)  # 96 - 48

        self.dense3_1 = RDB(self.nFeat * 4, self.nDenselayer, self.growthRate)
        self.dense3_2 = RDB(self.nFeat * 4, self.nDenselayer, self.growthRate)
        self.d_ch3 = nn.Conv2d(self.nFeat * 4, self.nFeat * 8, kernel_size=1, padding=0, bias=True)

        self.down4 = nn.MaxPool2d(2)  # 48 - 24

        self.dense4_1 = RDB(self.nFeat * 8, self.nDenselayer, self.growthRate)
        self.dense4_2 = RDB(self.nFeat * 8, self.nDenselayer, self.growthRate)
        self.d_ch4 = nn.Conv2d(self.nFeat * 8, self.nFeat * 8, kernel_size=1, padding=0, bias=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.output = nn.Sequential(
            nn.Conv2d(self.nFeat * 8, self.nFeat * 4, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.nFeat * 4, self.nFeat * 2, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.nFeat * 2, self.nChannel_out, kernel_size=1, padding=0, bias=True)
        )


    def forward(self, input_tensor):
        feat1 = F.leaky_relu(self.conv1(input_tensor))
        feat2 = F.leaky_relu(self.conv2(feat1))

        # downsampling
        down1 = self.down1(feat2)

        # dense blocks
        dfeat1_1 = self.dense1_1(down1)
        dfeat1_2 = self.dense1_2(dfeat1_1)
        bdown1 = self.d_ch1(dfeat1_2)

        # downsampling
        down2 = self.down2(bdown1)

        dfeat2_1 = self.dense2_1(down2)
        dfeat2_2 = self.dense2_2(dfeat2_1)
        bdown2 = self.d_ch2(dfeat2_2)

        # downsampling
        down3 = self.down3(bdown2)

        dfeat3_1 = self.dense3_1(down3)
        dfeat3_2 = self.dense3_2(dfeat3_1)
        bdown3 = self.d_ch3(dfeat3_2)

        # downsampling
        down4 = self.down4(bdown3)

        dfeat4_1 = self.dense4_1(down4)
        dfeat4_2 = self.dense4_2(dfeat4_1)
        bdown4 = self.d_ch4(dfeat4_2)

        # print(bdown4.shape)
        pooled = self.avg_pool(bdown4)
        output = self.output(pooled)

        b, c, h, w = output.shape
        output = torch.reshape(output, (b, c//3, 3, h, w))

        return output

if __name__ == '__main__':
    net = IlluminationModule()
    tensor = torch.randn(1, 3, 480, 640)
    out = net(tensor)


