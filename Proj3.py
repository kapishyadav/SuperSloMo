import os
import imageio
import torch
import torchvision
import cv2
import numpy as np
import pickle
from UNet import UNet
import torchvision.transforms as transforms
from UNetLoss import perceptual_loss, warping_loss
import random
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision.models as models

import sys

# torch.cuda.empty_cache()

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


# dataset_path = "/Users/nkroeger/Documents/UF_Grad/2020\ Fall/DL4CG/Part3/SuperSloMo/original_high_fps_videos/"
random.seed(0)

def my_test_train_split(filenames,train_perc):
	random.shuffle(filenames)
	train_len = int(train_perc*0.01*len(filenames))
	train_names = filenames[:train_len]
	test_names = filenames[train_len:]
	return train_len, train_names, test_names


# data_path = "ExtractedImages/"
# filenames = os.listdir(data_path)

# train_len, train_files, test_files = my_test_train_split(filenames, 80)


# def save_train(train_files, test_files, train_len):
# 	train = []
# 	print('split done')
# 	for file in train_files:
# 		loc = data_path+file+"/"
# 		i=0
# 		for image_path in os.listdir(loc):
# 			i=i+1
# 			image = Image.open(loc+image_path)
# 			image = ToTensor()(image)
# 			train.append(image)

# 		print(file)
# 	train = np.array(train)
# 	np.save("train.npy", train)
# 	print("\n\nTrain data saved\n")
# save_train(train_files, test_files, train_len)
train = []
folder = "ExtractedImages/IMG_0187.MOV/"
for i in range(0, 8):
	file = folder+"000"+str(i+1)+"0.jpg"
	img = Image.open(file)
	img = ToTensor()(img)
	train.append(img)



def resize_tensor(input_tensors, h, w):
	final_output = None
	batch_size, channel, height, width = input_tensors.shape
	input_tensors = torch.squeeze(input_tensors, 1)

	for img in input_tensors:
		img_PIL = transforms.ToPILImage()(img)
		img_PIL = torchvision.transforms.Resize([h,w])(img_PIL)
		img_PIL = torchvision.transforms.ToTensor()(img_PIL)
		if final_output is None:
			final_output = img_PIL
		else:
			final_output = torch.cat((final_output, img_PIL), 0)
		final_output = torch.unsqueeze(final_output, 1)
	return final_output




#Read in 2 images for initial results
img0 = resize_tensor(torch.unsqueeze(train[0], dim=0), 256, 256)
img1 = resize_tensor(torch.unsqueeze(train[-1], dim=0), 256, 256)

intermediateFrames = train[1:7]

# img0 = torch.from_numpy(np.asarray(cv2.imread('00205.jpg'))).float()
# img1 = torch.from_numpy(np.asarray(cv2.imread('00234.jpg'))).float()
#Normalize
# img0 = img0.permute(2, 0, 1)*2.5/255 + 0.01
# img1 = img1.permute(2, 0, 1)*2.5/255 + 0.01
#Resize
# img0 = resize_tensor(torch.unsqueeze(img0, dim = 0), 512, 512)
# img1 = resize_tensor(torch.unsqueeze(img1, dim = 0), 512, 512)


def backwarp(image, flow, device):
	#Use a flow to warp one image to another.
	#Implement the I = g(I, F) function
	image = image.cpu().detach().numpy()
	flow = flow.cpu().detach().numpy()
	flow = flow.squeeze()
	flowX = flow[0,:,:]
	flowY = flow[1,:,:]
	warpedImg = cv2.remap(image.squeeze().transpose(1,2,0), flowX, flowY, cv2.INTER_LINEAR)
	return torch.from_numpy(warpedImg).permute(2,0,1).unsqueeze(0).to(device)


###################
# Define Network
###################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

InputChannels_flowComp  = 6 # 6 input channels, for stacking 2 RGB images as input
OutputChannels_flowComp = 4 # 4 = 2 for each flow e.g. 0->1 or 1->0, where each flow has horizontal and vertical component
flowCompNet = UNet(InputChannels_flowComp, OutputChannels_flowComp)

InputChannels_arbFlow  = 20 # 20 = 6 RGB channels + 4 Flows + 6 backwarped images + 4 Flows from prev net
OutputChannels_arbFlow = 5 # 5 = 4 Flows (2channels each) + 1 channel Visibility map
arbFlowInterp = UNet(InputChannels_arbFlow, OutputChannels_arbFlow)

lmbda_r = 0.8
lmbda_p = 0.005
lmbda_w = 0.4
lmbda_s = 1

L1loss = torch.nn.L1Loss()

vgg16Model 		   = models.vgg16(pretrained=True)
Conv4_3Weights 	   = list(vgg16Model.children())[0][:22]
vgg16Model_Conv4_3 = torch.nn.Sequential(*Conv4_3Weights)

vgg16Model_Conv4_3.to(device)

for layer in vgg16Model_Conv4_3.parameters():
	layer.requires_grad = False

###################################
#Setting up training parameters
###################################
BothModelWeights = list(flowCompNet.parameters()) + list(arbFlowInterp.parameters())
optimizer = torch.optim.Adam(BothModelWeights, lr=0.01)

if torch.cuda.is_available():
	flowCompNet   = flowCompNet.cuda()
	arbFlowInterp = arbFlowInterp.cuda()




##############################
# Define Training Algorithm
##############################
train_loss = []
NumIntermediateFrames = 6
epochs     = 100
for e in range(0, epochs):

	print("epoch: ", e+1)

	#TODO: Get 2 frames and intermediate frames
	#Define input to 1st network
	img0 = img0.to(device)
	img1 = img1.to(device)
	ImageInput = torch.cat([img0, img1], dim=1)

	# ImageInput = torch.unsqueeze(ImageInput, dim = 0)
	#forward pass through 1st network
	import pdb;pdb.set_trace()

	Flows      = flowCompNet(ImageInput)
	Flow0to1   = Flows[:,0:2,:,:]
	Flow1to0   = Flows[:,2:,:,:]

	#Define input to 2nd network
	rLoss, pLoss, wLoss_2 = 0.0, 0.0, 0.0
	for i in range(0, NumIntermediateFrames):
		t=(i+1)/(NumIntermediateFrames+1)
		I_true_t =  resize_tensor(torch.unsqueeze(intermediateFrames[i], dim=0).to(device), 256, 256) #groundtruth intermediate frame
		#Calculate backwarp g() twice for t->I1 and t->I0 (3 channels for each)

		#Calculate F_hat from equation (4)
		F_hat_tto0 = -(1-t)*t*Flow0to1 + t*t*Flow1to0
		F_hat_tto1 = (1-t)*(1-t)* Flow0to1 - t*(1-t)*Flow1to0

		g0 = backwarp(img0, F_hat_tto0, device)
		g1 = backwarp(img1, F_hat_tto1, device)

		# Input to network 2 - arb time flow interpolation
		# 20 = 6 RGB channels + 4 Flows + 6 backwarped images + 4 Flows from prev net
		arbFlowInput = torch.cat([ImageInput, g1, F_hat_tto1, F_hat_tto0, g0, Flow0to1, Flow1to0], dim=1)

		#forward pass through 2nd network
		arbFlowOutputs = arbFlowInterp(arbFlowInput)

		V_t1    = arbFlowOutputs[:,  0,:,:]
		DeltaF_t1 = arbFlowOutputs[:,1:3,:,:]
		DeltaF_t0 = arbFlowOutputs[:,3: ,:,:]
		V_t0	  = 1 - V_t1 #equation (5)

		#Calculate predicted I_t using equation (1.5) from the paper
		F_tto0 = F_hat_tto0 + DeltaF_t0
		F_tto1 = F_hat_tto1 + DeltaF_t1

		Z = (1-t)*V_t0 + t*V_t1
		I_tHat = ((1-t)*V_t0*backwarp(img0, F_tto0, device) + t*V_t1*backwarp(img1, F_tto1, device))/Z
		
		# import pdb; pdb.set_trace()

		rLoss += L1loss(I_tHat, I_true_t)
		pLoss += perceptual_loss(I_tHat, I_true_t, vgg16Model_Conv4_3)
		wLoss_2+= L1loss(I_true_t, backwarp(img0, F_hat_tto0, device))
		wLoss_2+= L1loss(I_true_t, backwarp(img1, F_hat_tto1, device))
		local_vars = list(locals().items())
		for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()), key= lambda x: -x[1])[:10]:
			print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
		break
		

	wLoss = wLoss_2/NumIntermediateFrames + warping_loss(img0, img1, backwarp(i1,Flow0to1, device), backwarp(i0,Flow1to0, device)) 

	gradF_0to1 = L1loss(Flow0to1[:,:,:-1,:], Flow0to1[:,:,1:,:]) + L1loss(Flow0to1[:,:,:,:-1], Flow0to1[:,:,:,1:])
	gradF_1to0 = L1loss(Flow1to0[:,:,:-1,:], Flow1to0[:,:,1:,:]) + L1loss(Flow1to0[:,:,:,:-1], Flow1to0[:,:,:,1:])
	sLoss = gradF_0to1 + gradF_1to0
	#Calculate the losses with weights equation (7)
	
	predError = lmbda_r*(rLoss/NumIntermediateFrames) + lmbda_p*(pLoss/NumIntermediateFrames) + lmbda_w*wLoss + lmbda_s*sLoss

	#Update the weights
	predError.backward()
	optimizer.step()


	


#TODO: Given a 30fps video, predict intermediate frames and save out a 240fps video
