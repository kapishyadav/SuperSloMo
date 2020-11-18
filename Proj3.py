import os
import imageio
import torch
import torchvision
import numpy as np
import pickle

dataset_path = "DeepVideoDeblurring_Dataset_Original_High_FPS_Videos/original_high_fps_videos/"
filenames = os.listdir(dataset_path)

# videos = dict()
# numVideos = torch.zeros(len(filenames))
# count = 0
# for i in range(0,5):
# 	frames = []
# 	video = imageio.get_reader(dataset_path+filenames[i])
# 	for i, frame in enumerate(video):
# 		frames.append(frame)
# 	videos[count] = frames
# 	count = count+1
# 	print(count)
# with open('videos5.pickle', 'wb') as handle:
#     pickle.dump(videos, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("videos5.pickle","rb") as handle:
    videos = pickle.load(handle)

# print(len(videos[2]))



