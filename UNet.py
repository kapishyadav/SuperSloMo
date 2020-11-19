# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, LeakyReLU, ReLU, CrossEntropyLoss, ConvTranspose2d, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, UpsamplingNearest2d, BatchNorm2d
from torch.optim import Adam, SGD
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
            nn.AvgPool2d(stride=2)
            nn.Conv2d(32, 64, kernel_size=5, padding=1, stride =1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=1, stride =1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.encoder3 = nn.Sequential(
            nn.AvgPool2d(stride=2)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride =1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride =1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.encoder4 = nn.Sequential(
            nn.AvgPool2d(stride=2)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.encoder5 = nn.Sequential(
            nn.AvgPool2d(stride=2)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.encoder6 = nn.Sequential(
            nn.AvgPool2d(stride=2)
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
        # x = self.cnn_layers(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        upe6 = upsample(e6)
        cat1 = torch.cat([upe6, e5], dim=1)
        d1 = self.decoder1(cat1)
        upd1 = upsample(d1)
        cat2 = torch.cat([upd1, e4], dim=1)
        d2 = self.decoder2(cat2)
        upd2 = upsample(d2)
        cat3 = torch.cat([upd2, e3], dim=1)
        d3 = self.decoder3(cat3)
        upd3 = upsample(d3)
        cat4 = torch.cat([upd3, e2], dim=1)
        d4 = self.decoder4(cat4)
        upd4 = upsample(d4)
        cat5 = torch.cat([upd4, e1], dim=1)
        d5 = self.decoder5(cat5)
        conv_out = self.convout(d5)

        return conv_out
