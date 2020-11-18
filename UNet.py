# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, LeakyReLU, ReLU, CrossEntropyLoss, ConvTranspose2d, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, UpsamplingNearest2d, BatchNorm2d
from torch.optim import Adam, SGD
from torch.nn.functional import interpolate 
from torch import tanh

#Convert grayscale to LAB
class ColorNet(Module):
    def __init__(self):
        super(ColorNet, self).__init__()

        # Defining a 2D convolution layer 128*128
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3, stride =1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=7, padding=3, stride =1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True)
            )

        self.encoder2 = nn.Sequential(
            nn.AvgPool2d(stride=2)
            nn.Conv2d(32, 64, kernel_size=5, padding=1, stride =1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=5, padding=1, stride =1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.encoder3 = nn.Sequential(
            nn.AvgPool2d(stride=2)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride =1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride =1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.encoder4 = nn.Sequential(
            nn.AvgPool2d(stride=2)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.encoder5 = nn.Sequential(
            nn.AvgPool2d(stride=2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True))

        self.encoder6 = nn.Sequential(
            nn.AvgPool2d(stride=2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True))

        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            nn.Conv2d(2*512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True))

        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            nn.Conv2d(2*512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.decoder3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            nn.Conv2d(2*256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.decoder4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            nn.Conv2d(2*128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.decoder5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            nn.Conv2d(2*64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

        self.convout = nn.Sequential(
            nn.Conv2d(32, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(negative_slope = 0.1, inplace=True))

    # Defining the forward pass
    def forward(self, x):
        # x = self.cnn_layers(x)
        e1 = self.encoder1(x)
        e2 = self.encoder1(e1)
        e3 = self.encoder1(e2)
        e4 = self.encoder1(e3)
        e5 = self.encoder1(e4)
        e6 = self.encoder1(e5)
        d1 = self.decoder1(e6)
        cat1 = torch.cat([e5,d1], dim=1)



        return u5
