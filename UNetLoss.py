import torch
from torch.autograd import Variable
from torch.nn import Linear, LeakyReLU, MSELoss, ReLU, CrossEntropyLoss, ConvTranspose2d, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, UpsamplingNearest2d, BatchNorm2d
from torch.optim import Adam, SGD
from torch.nn.functional import interpolate
from torch import tanh
import torchvision.models as models

import numpy as np



def reconstruction_loss(i0, i1, ogSet, predSet, device):
	#Equation (8)
	N = len(ogSet)
	rl = (1/N) * (np.sum(np.linalg.norm((predSet - ogSet), ord=1)))
	return rl

def perceptual_loss(i0, i1, ogSet, predSet, device):
	#Equation (9)
	N = len(ogSet)

	vgg16Model 		   = models.vgg16(pretrained=True)
	Conv4_3Weights 	   = list(vgg16.children())[0][:22]
	vgg16Model_Conv4_3 = nn.Sequential(*Conv4_3Weights)

	vgg16Model_Conv4_3.to(device)

	for layer in vgg16Model_Conv4_3.parameters():
		layer.requires_grad = False

	ogSetPred  = vgg16Model_Conv4_3(ogSet)
	predSetOut = vgg16Model_Conv4_3(predSet)

	pl = MSELoss(ogSetPred, predSetOut)
	return pl

def warping_loss(i0, i1, ogSet, predSet, device):
	pass

def smoothness_loss():
	pass
