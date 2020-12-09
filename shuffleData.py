import torch
import os
from PIL import Image
from torchvision.transforms import ToTensor



def getData():
	#x1 = (I0, .... intermediate frames ..., I1) - called ImageGroup
	data_set = torch.zeros(3325, 8, 3, 128, 128) #3325 = (200*133)/8 frames
	input_path = "ExtractedImages/"
	all_videos = os.listdir(input_path)
	dataCounter = 0
	for video in all_videos:
		print(video)
		frames = os.listdir(input_path+video)
		numFrames = len(frames)
		for i in range(0, numFrames, 8):
			imageGroup = readListOfImages(input_path+video, frames[i:(i+8)])
			for j in range(8):
				data_set[dataCounter, j, :, :, :] = imageGroup[j]
			dataCounter += 1
	return data_set

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
	train_data = data[:int(0.9*data.shape[0]),:,:,:,:]
	test_data = data[int(0.9*data.shape[0]):,:,:,:,:]
	#split data into blocks, put each block into dataBatched
	train_dataChunk = torch.chunk(train_data.half(), int(train_data.shape[0]/batchSize) , dim=0)
	test_dataChunk  = torch.chunk(test_data.half(), int(test_data.shape[0]/batchSize) , dim=0)
	return train_dataChunk, test_dataChunk


batch_size = 2
print("Making data")
data_set = getData()
print("Shuffling and batching train, test data")
train_set, test_set = shuffleDataAndBatch(data_set, batch_size)
torch.save(train_set, "train_set_bSize"+str(batch_size)+".pt")
torch.save(test_set, "test_set_bSize"+str(batch_size)+".pt")
print("saved")
