import os
import imageio
import torch
import torchvision
import cv2
import numpy as np
import pickle
from UNet import UNet
import torchvision.transforms as transforms
from UNetLoss import reconstruction_loss, perceptual_loss, warping_loss, smoothness_loss
import random
from PIL import Image
from torchvision.transforms import ToTensor

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
img0 = torch.from_numpy(np.asarray(cv2.imread('00205.jpg'))).float()
img1 = torch.from_numpy(np.asarray(cv2.imread('00234.jpg'))).float()
#Normalize
img0 = img0.permute(2, 0, 1)*2.5/255 + 0.01
img1 = img1.permute(2, 0, 1)*2.5/255 + 0.01
#Resize
img0 = resize_tensor(torch.unsqueeze(img0, dim = 0), 512, 512)
img1 = resize_tensor(torch.unsqueeze(img1, dim = 0), 512, 512)


def backwarp(image, flow):
	#Use a flow to warp one image to another.
	#Implement the I = g(I, F) function
	flow = flow.squeeze()
	flowX = flow[0,:,:]
	flowY = flow[1,:,:]
	warpedImg = cv2.remap(image.squeeze().transpose(1,2,0), flowX, flowY, cv2.INTER_LINEAR)
	return warpedImg


###################
# Define Network
###################
InputChannels_flowComp  = 6 # 6 input channels, for stacking 2 RGB images as input
OutputChannels_flowComp = 4 # 4 = 2 for each flow e.g. 0->1 or 1->0, where each flow has horizontal and vertical component
flowCompNet = UNet(InputChannels_flowComp, OutputChannels_flowComp)

InputChannels_arbFlow  = 20 # 20 = 6 RGB channels + 4 Flows + 6 backwarped images + 4 Flows from prev net
OutputChannels_arbFlow = 5 # 5 = 4 Flows (2channels each) + 1 channel Visibility map
arbFlowInterp = UNet(InputChannels_arbFlow, OutputChannels_arbFlow)

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
for i in range(0, epochs):
	print("epoch: ", i+1)

	#TODO: Get 2 frames and intermediate frames
	#Define input to 1st network
	ImageInput = torch.cat([img0, img1], dim=0).permute(1,0,2,3)
	# ImageInput = torch.unsqueeze(ImageInput, dim = 0)
	#forward pass through 1st network
	Flows      = flowCompNet(ImageInput)
	Flow0to1   = Flows[:,0:2,:,:]
	Flow1to0   = Flows[:,2:,:,:]

	#Define input to 2nd network
	rLoss, pLoss = 0.0, 0.0
	for i in range(1,NumIntermediateFrames+1):
		t=i/(NumIntermediateFrames+1)
		# I_true_t =  Images[t] #groundtruth intermediate frame
		#Calculate backwarp g() twice for t->I1 and t->I0 (3 channels for each)
		img0_np     = img0.detach().numpy()
		Flow0to1_np = Flow0to1.detach().numpy()
		img1_np     = img1.detach().numpy()
		Flow1to0_np = Flow1to0.detach().numpy()

		#Calculate F_hat from equation (4)
		F_hat_tto0 = -(1-t)*t*Flow0to1_np + t*t*Flow1to0_np
		F_hat_tto1 = (1-t)*(1-t)* Flow0to1_np - t*(1-t)*Flow1to0_np

		g0 = torch.from_numpy(backwarp(img0_np, F_hat_tto0)).permute(2,0,1).unsqueeze(0)
		g1 = torch.from_numpy(backwarp(img1_np, F_hat_tto1)).permute(2,0,1).unsqueeze(0)

		# Input to network 2 - arb time flow interpolation
		# 20 = 6 RGB channels + 4 Flows + 6 backwarped images + 4 Flows from prev net
		arbFlowInput = torch.cat([ImageInput, g1, torch.from_numpy(F_hat_tto1), torch.from_numpy(F_hat_tto0), g0, Flow0to1, Flow1to0], dim=1)

		#forward pass through 2nd network
		arbFlowOutputs = arbFlowInterp(arbFlowInput)

		V_t1    = arbFlowOutputs[:,  0,:,:]
		DeltaF_t1 = arbFlowOutputs[:,1:3,:,:]
		DeltaF_t0 = arbFlowOutputs[:,3: ,:,:]
		V_t0	  = 1 - V_t1 #equation (5)

		#Calculate predicted I_t using equation (1.5) from the paper
		Z = (1-t)*V_t0 + t*V_t1
		I_tHat = ((1-t)*V_t0*g0 + t*V_t1*g1)/Z
		import pdb; pdb.set_trace()

		rLoss += np.linalg.norm((I_tHat - I_true_t), ord=1)
		pLoss += perceptual_loss(I_tHat, I_t)
	#Calculate the losses with weights equation (7)
	predError = rLoss/NumIntermediateFrames + pLoss/NumIntermediateFrames + wLoss + sLoss

	#Update the weights
	predError.backward()
	optimizer.step()


#TODO: Given a 30fps video, predict intermediate frames and save out a 240fps video
