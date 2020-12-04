import torch
from torch.autograd import Variable
from torch.nn import Linear, LeakyReLU, MSELoss, ReLU, CrossEntropyLoss, ConvTranspose2d, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, UpsamplingNearest2d, BatchNorm2d
from torch.optim import Adam, SGD
from torch.nn.functional import interpolate
from torch import tanh
import torchvision.models as models
import numpy as np



# def reconstruction_loss(ogSet, predSet):
# 	#Equation (8)
# 	N = len(ogSet)
# 	rl = (1/N) * (np.sum(np.linalg.norm((predSet - ogSet), ord=1)))
# 	return rl

def perceptual_loss(I_tHat, I_t):
	#Equation (9)
	vgg16Model 		   = models.vgg16(pretrained=True)
	Conv4_3Weights 	   = list(vgg16.children())[0][:22]
	vgg16Model_Conv4_3 = nn.Sequential(*Conv4_3Weights)

	vgg16Model_Conv4_3.to(device)

	for layer in vgg16Model_Conv4_3.parameters():
		layer.requires_grad = False

	ogSetPred  = vgg16Model_Conv4_3(I_t)
	predSetOut = vgg16Model_Conv4_3(I_tHat)

	pl = torch.linalg.norm(predSet - ogSet, p=2)
	return pl

def warping_loss(i0, i1, g_i1_F01, g_i0_F10):
	lw = torch.linalg.norm(i0 - g_i1_F01 , p=1) + torch.linalg.norm(i1 - g_i0_F10, p=1)
	return lw

