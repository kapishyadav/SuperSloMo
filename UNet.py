# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, LeakyReLU, ReLU, CrossEntropyLoss, ConvTranspose2d, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, UpsamplingNearest2d, BatchNorm2d
from torch.optim import Adam, SGD
import torch.nn as nn
from torch.nn.functional import interpolate
from torch import tanh

#Convert grayscale to LAB
class UNet(Module):
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()

        # Defining a 2D convolution layer 128*128
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, padding=3, stride =1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            nn.Conv2d(32, 32, kernel_size=7, padding=3, stride =1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True)
            )

        self.encoder2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=3, stride =1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=1, stride =1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.encoder3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride =1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride =1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.encoder4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.encoder5 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.encoder6 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True))

        self.decoder1 = nn.Sequential(
            nn.Conv2d(2*512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True))

        self.decoder2 = nn.Sequential(
            nn.Conv2d(2*256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.decoder3 = nn.Sequential(
            nn.Conv2d(2*128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.decoder4 = nn.Sequential(
            nn.Conv2d(2*64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.decoder5 = nn.Sequential(
            nn.Conv2d(2*32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.convout = nn.Sequential(
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

    # Defining the forward pass
    def forward(self, x):
        print("x",x.shape)
        e1 = self.encoder1(x)
        print("e1",e1.shape)
        e2 = self.encoder2(e1)
        print("e2",e2.shape)
        e3 = self.encoder3(e2)
        print("e3",e3.shape)
        e4 = self.encoder4(e3)
        print("e4",e4.shape)
        e5 = self.encoder5(e4)
        print("e5",e5.shape)
        e6 = self.encoder6(e5)
        print("e6",e6.shape)
        upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        upe6 = upsample(e6)
        print("upe6",upe6.shape)

        cat1 = torch.cat([upe6, e5], dim=1)
        print("cat1", cat1.shape)
        d1 = self.decoder1(cat1)
        print("d1",d1.shape)
        upd1 = upsample(d1)
        print("upd1",upd1.shape)
        cat2 = torch.cat([upd1, e4], dim=1)
        print("cat2",cat2.shape)
        d2 = self.decoder2(cat2)
        print("d2",d2.shape)
        upd2 = upsample(d2)
        print("upd2",upd2.shape)
        cat3 = torch.cat([upd2, e3], dim=1)
        print("cat3",cat3.shape)
        d3 = self.decoder3(cat3)
        print("d3",d3.shape)
        upd3 = upsample(d3)
        print("upd3",upd3.shape)
        cat4 = torch.cat([upd3, e2], dim=1)
        print("cat4",cat4.shape)
        d4 = self.decoder4(cat4)
        print("d4",d4.shape)
        upd4 = upsample(d4)
        print("upd4",upd4.shape)
        cat5 = torch.cat([upd4, e1], dim=1)
        print("cat5",cat5.shape)
        d5 = self.decoder5(cat5)
        print("d5",d5.shape)
        conv_out = self.convout(d5)
        print("conv_out",conv_out.shape)

        return conv_out
