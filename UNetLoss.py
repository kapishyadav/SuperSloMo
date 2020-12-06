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
mse = torch.nn.MSELoss()
L1loss = torch.nn.L1Loss()
def perceptual_loss(I_tHat, I_true_t, vgg16Model_Conv4_3):
	#Equation (9)
	I_true_t_features  = vgg16Model_Conv4_3(I_true_t)
	I_tHat_features = vgg16Model_Conv4_3(I_tHat)

	pl = mse(I_tHat_features, I_true_t_features)
	return pl

def warping_loss(i0, i1, g_i1_F01, g_i0_F10):
	lw = L1loss(i0 , g_i1_F01) + L1loss(i1, g_i0_F10)
	return lw

