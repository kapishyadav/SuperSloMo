import torch
from torch.autograd import Variable
from torch.nn import Linear, LeakyReLU, MSELoss, ReLU, CrossEntropyLoss, ConvTranspose2d, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, UpsamplingNearest2d, BatchNorm2d
from torch.optim import Adam, SGD
from torch.nn.functional import interpolate 
from torch import tanh
import torchvision.models as models

import numpy as np



def reconstruction_loss(i0, i1, ogSet, predSet, device):
	N = len(ogSet)
	lr = (1/N) * (np.sum(np.linalg.norm((predSet - ogSet), ord=1)))
	return lr

def perceptual_loss(i0, i1, ogSet, predSet, device):
	N = len(ogSet)

	vgg16 = models.vgg16(pretrained=True)
	vgg16_conv4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
	vgg16_conv4_3.to(device)
	for layer in vgg16_conv_4_3.parameters():
		layer.requires_grad = False

	lp = MSELoss(vgg16_conv4_3(ogSet), vgg16_conv4_3(predSet))
	return lp

def warping_loss(i0, i1, ogSet, predSet, device):
	pass

def smoothness_loss:
	pass

