import cv2
import time
from faceclass import face_detection
#foo = SourceFileLoader("faceclass.py", "/anyaranavat/Downloads/faceid/faceclass.py").load_module()
#from faceclass import face_detection
from camera import Camera
import os
import glob
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision
import torch
from torch.optim import Adam, SGD
from torch.nn import Linear, ReLU, Sequential, MaxPool2d, Conv2d, Module, Softmax, CrossEntropyLoss
import numpy as np
class CNN(Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()

        # output size of conv2d layer is ((w-f+2P)/s)+1

        # input size if (16,3,128,128)
        self.cnn_layers = Sequential(
            Conv2d(in_channels=3, out_channels=80, kernel_size=5),
            # output = (16,32,128,128)
            ReLU(),
            Conv2d(in_channels=80, out_channels=80, kernel_size=5),
            # output = (16,32,128,128)
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        # output = (16,32,64,64)
        self.fc1 = Linear(80*85*85, num_classes)

    def forward(self, x):
        out = self.cnn_layers(x)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        return out