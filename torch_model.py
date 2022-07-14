import torch
from torch import nn, optim
from torch.nn import Linear, ReLU, Sequential, MaxPool2d, Conv2d, Module, Softmax
from torch.optim import Adam, SGD
from torchvision.transforms import transforms
import numpy as np
import cv2
import os
from torch.utils.data import random_split

class ImageReader:
	
	def __init__(self,dataset_path):
		self.dataset_path = dataset_path
		#self.data=

	def readImage(self,image):
		image_path = self.dataset_path + "/" + image
		return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
 
	def loadAndReSizeImage(self, image):
		img = self.readImage(image)
		resized = cv2.resize(img,(128,128))
		return resized

	def mapImagesToTensor(self):
		images = map(self.loadAndReSizeImage,os.listdir(self.dataset_path))
		images = np.array(list(images))
		img_tensors = torch.from_numpy(images)
		return img_tensors


class CNN(Module):
	def __init__(self,num_classes = 2):
		super(CNN, self).__init__()
	
		self.cnn_layers = Sequential(
				Conv2d(in_channels = 3 , out_channels =32 ,kernel_size=5),
				ReLU(),
				Conv2d(in_channels = 32, out_channels = 32, kernel_size = 5),
				ReLU(),
				MaxPool2d(kernel_size = 2, stride=2))
		self.fc1 = Linear(1600,128)
		self.relu = ReLU()
		self.fc2 = Linear(128,2)
 
	def forward(self,x):
		out = self.cnn_layers(x)
		out = self.fc1(out)
		out = self.relu(out)
		out = self.fc2(out)
		return out
 
