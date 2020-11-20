import os
import imageio
import torch
import torchvision
import numpy as np
import pickle
from UNet import UNet
from UNetLoss import reconstruction_loss, perceptual_loss, warping_loss, smoothness_loss

dataset_path = "/Users/nkroeger/Documents/UF_Grad/2020\ Fall/DL4CG/Part3/SuperSloMo/original_high_fps_videos/"
filenames = os.listdir(dataset_path)

#Load in dataset
# videos = dict()
# numVideos = torch.zeros(len(filenames))
# count = 0
# for i in range(0,2):
# 	frames = []
# 	video = imageio.get_reader(dataset_path+filenames[i])
# 	for i, frame in enumerate(video):
# 		frames.append(frame)
# 	videos[count] = frames
# 	count = count+1
# 	print(count)
# with open('videos2.pickle', 'wb') as handle:
#     pickle.dump(videos, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open("videos5.pickle","rb") as handle:
#     videos = pickle.load(handle)

# print(len(videos[2]))


##############################
# Define Training Algorithm
##############################
def train_Reg(ConvModel, optimizer, loss, L_channel, a_b_average):

	tr_loss = 0

	if torch.cuda.is_available():
		L_channel = L_channel.cuda()
		a_b_average = a_b_average.cuda()

	optimizer.zero_grad()
	pred = ConvModel(L_channel)

	predError = loss(pred, a_b_average)
	predError.backward()
	optimizer.step()

	tr_loss = predError.item()
	print("Loss: " ,tr_loss)
	train_loss.append(tr_loss)

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

train_loss = []
epochs     = 100

##############################
# Define Training Algorithm
##############################
for i in range(0, epochs):
    print("epoch: ", i+1)
    # train_Reg(ConvModel, optimizer, loss, L_channel, a_b_average)

    #TODO: Get 2 frames and intermediate frames


    #Define input to 1st network
    ImageInput = torch.cat([Frame0, Frame1], dim=1)
    #forward pass through 1st network
    Flows      = flowCompNet(ImageInput)
    Flow0to1   = Flows[:,0:2,:,:]
    Flow1to0   = Flows[:,2:,:,:]

    #Define input to 2nd network

    #Calculate backwarp twice for t->I1 and t->I0 (3 channels for each)

    #Calculate F_hat from equation (4)

    #forward pass through 2nd network
    arbFlowOutputs = arbFlowInterp()
    V_t1      = arbFlowOutputs[:,0,  :,:]
    DeltaF_t1 = arbFlowOutputs[:,1:3,:,:]
    DeltaF_t0 = arbFlowOutputs[:,3: ,:,:]

    #Calculate predicted I_t using equation (1.5) from the paper
    I_tHat = ~

    #Calculate the losses
    reconstruction_loss, perceptual_loss, warping_loss, smoothness_loss

    predError = rLoss + pLoss + wLoss + sLoss

    #Update the weights
    predError.backward()
    optimizer.step()
