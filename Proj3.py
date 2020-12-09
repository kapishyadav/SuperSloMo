import os
import imageio
import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import pickle
from UNet import UNet
import torchvision.transforms as transforms
from UNetLoss import perceptual_loss, warping_loss
import random
from PIL import Image
import backwarp
from torchvision.transforms import ToTensor
import torchvision.models as models
from pympler import asizeof
import sys
import subprocess
import datetime

def get_gpu_memory_map():
    # Get the current gpu usage.
    # Returns
    # usage: dict - Keys are device ids as integers. Values are memory usage as integers in MB.
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

torch.cuda.empty_cache()

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

random.seed(0)

def my_test_train_split(filenames,train_perc):
    random.shuffle(filenames)
    train_len = int(train_perc*0.01*len(filenames))
    train_names = filenames[:train_len]
    test_names = filenames[train_len:]
    return train_len, train_names, test_names

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

###################
# Define Network
###################
now = datetime.datetime.now()
now = '%d%02d%02d_%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
loadModel = True
Height, Width = 128, 128
learningRate = 0.01

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

InputChannels_flowComp  = 6 # 6 input channels, for stacking 2 RGB images as input
OutputChannels_flowComp = 4 # 4 = 2 for each flow e.g. 0->1 or 1->0, where each flow has horizontal and vertical component
flowCompNet  = UNet(InputChannels_flowComp, OutputChannels_flowComp).to(device)
flowBackWarp = backwarp.backwarp(Height, Width, device).to(device)

InputChannels_arbFlow  = 20 # 20 = 6 RGB channels + 4 Flows + 6 backwarped images + 4 Flows from prev net
OutputChannels_arbFlow = 5 # 5 = 4 Flows (2channels each) + 1 channel Visibility map
arbFlowInterp = UNet(InputChannels_arbFlow, OutputChannels_arbFlow).to(device)

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
optimizer = torch.optim.Adam(BothModelWeights, lr=learningRate)

# Train Net #
def trainNet(ImageInput, intermediateFrames):
    Flows      = flowCompNet(ImageInput)
    Flow0to1   = Flows[:,0:2,:,:]
    Flow1to0   = Flows[:,2:,:,:]

    #Define input to 2nd network
    rLoss, pLoss, wLoss_2 = 0.0, 0.0, 0.0
    for i in range(0, NumIntermediateFrames):
        t = (i+1)/(NumIntermediateFrames+1)
        I_true_t = torch.squeeze(intermediateFrames[:,i,:,:,:]).to(device) #groundtruth intermediate frame
        #Calculate backwarp g() twice for t->I1 and t->I0 (3 channels for each)
        optimizer.zero_grad()
        #Calculate F_hat from equation (4)
        F_hat_tto0 = -(1-t)*t*Flow0to1 + t*t*Flow1to0
        F_hat_tto1 = (1-t)*(1-t)* Flow0to1 - t*(1-t)*Flow1to0
        g0 = flowBackWarp(img0, F_hat_tto0)
        g1 = flowBackWarp(img1, F_hat_tto1)
        # print("After g0,g1", get_gpu_memory_map())
        # Input to network 2 - arb time flow interpolation
        # 20 = 6 RGB channels + 4 Flows + 6 backwarped images + 4 Flows from prev net
        arbFlowInput = torch.cat([ImageInput, g1, F_hat_tto1, F_hat_tto0, g0, Flow0to1, Flow1to0], dim=1)
        #forward pass through 2nd network
        arbFlowOutputs = arbFlowInterp(arbFlowInput)
        # print("After arbFlowOutputs", get_gpu_memory_map())
        V_t1      = torch.unsqueeze(arbFlowOutputs[:,  0,:,:], dim = 1)
        DeltaF_t1 = arbFlowOutputs[:,1:3,:,:]
        DeltaF_t0 = arbFlowOutputs[:,3: ,:,:]
        V_t0	  = 1 - V_t1 #equation (5)

        #Calculate predicted I_t using equation (1.5) from the paper
        F_tto0 = F_hat_tto0 + DeltaF_t0
        F_tto1 = F_hat_tto1 + DeltaF_t1

        Z = (1-t)*V_t0 + t*V_t1
        #import pdb;pdb.set_trace()
        I_tHat = ((1-t)*V_t0*flowBackWarp(img0, F_tto0) + t*V_t1*flowBackWarp(img1, F_tto1))/Z
        # print("After I_tHat", get_gpu_memory_map())
        rLoss   += L1loss(I_tHat, I_true_t)
        pLoss   += perceptual_loss(I_tHat, I_true_t, vgg16Model_Conv4_3)
        wLoss_2 += L1loss(I_true_t, flowBackWarp(img0, F_hat_tto0))
        wLoss_2 += L1loss(I_true_t, flowBackWarp(img1, F_hat_tto1))

        # local_vars = list(locals().items())
        # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()), key= lambda x: -x[1])[:10]:
        # 	print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        # break

    wLoss = wLoss_2/NumIntermediateFrames + warping_loss(img0, img1, flowBackWarp(img1,Flow0to1), flowBackWarp(img0,Flow1to0))
    # print("After wLoss", get_gpu_memory_map())
    gradF_0to1 = L1loss(Flow0to1[:,:,:-1,:], Flow0to1[:,:,1:,:]) + L1loss(Flow0to1[:,:,:,:-1], Flow0to1[:,:,:,1:])
    gradF_1to0 = L1loss(Flow1to0[:,:,:-1,:], Flow1to0[:,:,1:,:]) + L1loss(Flow1to0[:,:,:,:-1], Flow1to0[:,:,:,1:])
    sLoss      = gradF_0to1 + gradF_1to0
    #Calculate the losses with weights equation (7)

    predError = lmbda_r*(rLoss/NumIntermediateFrames) + lmbda_p*(pLoss/NumIntermediateFrames) + lmbda_w*wLoss + lmbda_s*sLoss
    # print("pred error: ", predError.item())
    #Update the weights
    predError.backward()
    optimizer.step()
    # print("After optimizer", get_gpu_memory_map())
    return predError

##############################
# Define Training Algorithm
##############################
if loadModel:
    stateDict = torch.load('tlkasdhflaksjdf.pt')
    ArbFlowModel.load_state_dict(stateDict['ArbFlowModel'])
    FlowCompNet.load_state_dict(stateDict['FlowCompNet'])
    #OLD:
    # ArbFlowModel = torch.load('ArbFlowModel.pt', map_location=device)
    # FlowCompNet = torch.load('FlowCompNet.pt', map_location=device)
else:
    ModelPath = now+"/"
    os.makedirs(ModelPath, exist_ok=True)
    #Train network
    train_loss = []
    NumIntermediateFrames = 6
    epochs     = 2
    batch_size = 2
    train_set = torch.load("train_set_bSize"+str(batch_size)+".pt")
    test_set  = torch.load("test_set_bSize"+str(batch_size)+".pt")
    for e in range(0, epochs):
        print("epoch: ", e+1)
        for k in range(0, 5):#len(train_set)):
            batchLoss = 0.0
            currBatch = train_set[k].float() #shape: batchSize, 8, 3, Height, Width

            img0 = torch.squeeze(currBatch[:,0, :, :, :])
            img1 = torch.squeeze(currBatch[:,-1, :, :, :])

            intermediateFrames = currBatch[:, 1:(NumIntermediateFrames+1), :, :, :]

            #Define input to 1st network
            img0 = img0.to(device)
            img1 = img1.to(device)
            ImageInput = torch.cat([img0, img1], dim=1)
            batchLoss += trainNet(ImageInput, intermediateFrames).item()
            # print('Batch Loss:', batchLoss.item())
        train_loss.append(batchLoss/len(train_set))
        print('Train Loss:', train_loss[-1])

        #Save parameters and models
        SaveDict = {
            'epochs': epochs, 'batchSize': batch_size,
            'NumIntermediateFrames': NumIntermediateFrames,
            'learningRate': learningRate, 'train_loss': train_loss,
            'lmbda_r': lmbda_r, 'lmbda_p': lmbda_p, 'lmbda_w': lmbda_w, 'lmbda_s': lmbda_s,
            'flowComp': flowCompNet.state_dict(),
            'arbTimeInterp': arbFlowInterp.state_dict(),
        }
        torch.save(SaveDict, ModelPath+"SavedModel_epoch"+str(e)+".pt")


test_set     = torch.load("test_set_bSize2.pt")
tempBatchIdx = 0
tempBatch    = test_set[tempBatchIdx]

img0 = torch.squeeze(tempBatch[0, 0, :, :, :])
img1 = torch.squeeze(tempBatch[0,-1, :, :, :])

save_image(img0.float(), 'img0.png')
save_image(img1.float(), 'img1.png')

#Don't shuffle the test dataset. We want it to be a contiguous video. Rewrite some of the shuffleData.py

#TODO:
#DONESplit dataset into train/test
#DONEbackwarp function, remove cv2, use pytorch
#DONEtrain, save out trained model
#evaluation metrics for report
# e.g. training accuracy, reconstruction losses etc.
#Show the flow, visibility, I0, IIt and It_predicted

#Report, 2 videos

#TODO: Given a 30fps video, predict intermediate frames and save out a 240fps video


################
###Evaluation###
################

# def interpolationError(I_gt, I_inter):
# 	err = np.asarray(ImageChops.difference(I_gt))
# "...interpolation error (IE) [1], which
# is defined as root-mean-squared (RMS) difference between
# the ground-truth image and the interpolated image."
