import torch
import os
from PIL import Image
from torchvision.transforms import ToTensor



def getTrainingData():
	#x1 = (I0, .... intermediate frames ..., I1) - called ImageGroup
	train_set = torch.zeros(3325, 8, 3, 128, 128)
	input_path = "ExtractedImages/"
	all_videos = os.listdir(input_path)
	dataCounter = 0
	for video in all_videos:
		frames = os.listdir(input_path+video)
		numFrames = len(frames)
		for i in range(0, numFrames, 8):
			imageGroup = readListOfImages(input_path+video, frames[i:(i+8)])
			for j in range(8):
				train_set[dataCounter, j, :, :, :] = imageGroup[j]
			dataCounter += 1
	return train_set

def readListOfImages(videoPath, fileList):
	tempImages = []
	for file in fileList:
		img = Image.open(videoPath+"/"+file)
		img = ToTensor()(img)
		tempImages.append(img)
	return tempImages

def shuffleDataAndBatch(data, batchSize):
	#get random indices for shuffling data
	data = data[torch.randperm(data.shape[0]), :,:,:,:]
	#split data into blocks, put each block into dataBatched
	dataChunk = torch.chunk(data.half(), int(data.shape[0]/batchSize) , dim=0)
	return dataChunk


batch_size = 25
print("Making training data")
train_set = getTrainingData()
print("Shuffling and batching")
train_set = shuffleDataAndBatch(train_set, batch_size)
torch.save(train_set, "train_set_bSize"+str(batch_size)+".pt")
print("saved")
