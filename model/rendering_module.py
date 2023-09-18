import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import nn

from model.blocks import RDB


class RenderingModule(nn.Module):

    def __init__(self, channels_in=4, channels_out=16):
        super(RenderingModule, self).__init__()

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

        # self.transformer = TransformerModel(256, 6, 8, 512)

        # feature fusion bottleneck
        self.GFF_1x1_b = nn.Conv2d(self.nFeat * 8, self.nFeat * 8, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3_b = nn.Conv2d(self.nFeat * 8, self.nFeat * 8, kernel_size=3, padding=1, bias=True)

        # upsample
        self.up4 = nn.Upsample(scale_factor=2)  # 48 - 96

        self.GFF_1x1_5 = nn.Conv2d(self.nFeat * 8 * 2, self.nFeat * 8, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3_5 = nn.Conv2d(self.nFeat * 8, self.nFeat * 4, kernel_size=3, padding=1, bias=True)

        # upsample
        self.up5 = nn.Upsample(scale_factor=2)  # 96 - 192
        self.GFF_1x1_6 = nn.Conv2d(self.nFeat * 4 * 2, self.nFeat * 4, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3_6 = nn.Conv2d(self.nFeat * 4, self.nFeat * 2, kernel_size=3, padding=1, bias=True)

        self.up6 = nn.Upsample(scale_factor=2)  # 96 - 192
        self.GFF_1x1_7 = nn.Conv2d(self.nFeat * 2 * 2, self.nFeat * 2, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3_7 = nn.Conv2d(self.nFeat * 2, self.nFeat, kernel_size=3, padding=1, bias=True)

        # upsample
        self.up7 = nn.Upsample(scale_factor=2)  # 192 - 384
        self.top_feat = nn.Conv2d(self.nFeat * 2, self.nFeat, kernel_size=3, padding=1, bias=True)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(self.nFeat, self.nChannel_out, kernel_size=3, padding=1, bias=True),
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

        # b, c, h, w = down4.shape
        # new_t = torch.reshape(down4, [b, c, h * w])  # reshape image to sequence
        # new_t = new_t.permute(0, 2, 1)  # permute axis to make it fully convolutional

        # transformed = self.transformer(new_t)  # apply transformer
        # transformed = transformed.permute(0, 2, 1)  # restore axis post transformer

        # original shape
        # decoder_input = torch.reshape(transformed, [b, c, h, w])  # reshape to small image

        bff_3 = self.GFF_3x3_b(down4)
        # bottle neck dense dilated -------------------------------------------------------

        # upsample
        up4 = self.up4(bff_3)  # 96

        f_u4 = F.leaky_relu(torch.cat([up4, dfeat3_2, dfeat3_1], 1))
        ff_up4_1 = self.GFF_1x1_5(f_u4)
        ff_up4_2 = F.leaky_relu(self.GFF_3x3_5(ff_up4_1))

        up5 = self.up5(ff_up4_2)  # 192

        f_u5 = torch.cat([up5, dfeat2_2, dfeat2_1], 1)
        ff_up5_1 = F.leaky_relu(self.GFF_1x1_6(f_u5))
        ff_up5_2 = F.leaky_relu(self.GFF_3x3_6(ff_up5_1))

        up6 = self.up6(ff_up5_2)  # 384

        f_u6 = torch.cat([up6, dfeat1_2, dfeat1_1], 1)
        ff_up6_1 = F.leaky_relu(self.GFF_1x1_7(f_u6))
        ff_up6_2 = F.leaky_relu(self.GFF_3x3_7(ff_up6_1))

        up7 = self.up7(ff_up6_2)  # 384
        final_cat = torch.cat([up7, feat1], 1)
        top_f = F.leaky_relu(self.top_feat(final_cat))

        rgb = self.to_rgb(top_f)

        return rgb







if __name__ == '__main__':
    tensor = torch.randn(1, 3, 480, 640)
    net = RenderingModule(channels_in=3, channels_out=3)
    print('# Shading Net parameters:', sum(param.numel() for param in net.parameters()))

    out = net(tensor)
    print(out.shape)

