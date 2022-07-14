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
from PIL import Image
import numpy as np

#get the data path
cwd = os.getcwd()
dataset_path = os.path.join(cwd,"dataset")

#transform the data
transformer = transforms.Compose([
	transforms.Resize((178,178)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) #(0-255) to (-1,1)
])

#load the data with the above transformation
imageFolder = torchvision.datasets.ImageFolder(dataset_path, transform = transformer)
print("imageFolder: ", imageFolder)
n= len(imageFolder)
n_test=int(0.25*n)
test_set= torch.utils.data.Subset(imageFolder, range(n_test))
train_set= torch.utils.data.Subset(imageFolder, range(n_test, n))
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=True)
##train_loader = DataLoader(torchvision.datasets.ImageFolder(dataset_path, transform = transformer),batch_size = 16,shuffle = True)
##print("train_loader", train_loader)
##for i, (images,labels) in enumerate(train_loader):
##    print("i: " , i, "images: ", images, "labels: " , labels)
##    print("images size: " , images.size())
##x=np.random.randn(1000,2)
##y=np.random.randint(0,10,size=1000)
##
##x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.1,stratify=y)
##np.uniquely(y_train, return_counts=True)
##np.uniquely(y_val,return_counts=True)


#split the data in training and testing
#train_set,val_set=torch.utils.data.random_split(train_loader, [100,400])
#create the model class
class CNN(Module):
    def __init__(self,num_classes = 2):
        super(CNN, self).__init__()
        
        #output size of conv2d layer is ((w-f+2P)/s)+1

        #input size if (16,3,128,128)

        self.cnn_layers = Sequential(
                Conv2d(in_channels = 3 , out_channels =80,kernel_size=5),
            #output = (16,32,128,128)
                ReLU(),
                Conv2d(in_channels = 80, out_channels=80, kernel_size = 5),
            #output = (16,32,128,128)
                ReLU(),
                MaxPool2d(kernel_size = 2, stride=2))
        #output = (16,32,64,64)
        self.fc1 = Linear(80*85*85, num_classes)
 
    def forward(self,x):
        out = self.cnn_layers(x)
        #print(out.shape)
        out = torch.flatten(out, 1)
        #print(out.shape)
        out = self.fc1(out)
        #print(out.shape)
        return out

#create the model
model = CNN()
#output = model(images)

#optimizer and loss functions
optimizer = Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001 )
loss_function = CrossEntropyLoss()



num_epochs = 10
train_count = len(train_set)
test_count = len(test_set)
print("train count: ", train_count)
print("test count: ", test_count)
best_accuracy = 0

#train model
for epoch in range(num_epochs):

    train_accuracy = 0.0
    model.train()
    for i, (images,labels) in enumerate(train_loader):

        optimizer.zero_grad()
        outputs = model(images)
        #calculate the loss and back propogate
        loss = loss_function(outputs,labels)
        loss.backward()
        optimizer.step()

        _,prediction = torch.max(outputs.data,1)
        train_accuracy += int(torch.sum(prediction==labels.data))

    train_accuracy = train_accuracy/train_count
    print("TRAIN EPOCH: " + str(epoch) + " accuracy:  ", train_accuracy)
    
    model.eval()

    #test data
    test_accuracy = 0.0
    model.train()
    for i, (images,labels) in enumerate(test_loader):

        outputs = model(images)

        _,prediction = torch.max(outputs.data,1)
        test_accuracy += int(torch.sum(prediction==labels.data))

    test_accuracy = test_accuracy/test_count
    print("TEST EPOCH: " + str(epoch) + " accuracy:  ", test_accuracy)

    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), "best_checkout_model2.pth")
        best_accuracy = test_accuracy
    
# for pic in os.listdir("./images"):
#     path = os.path.join("images", pic)
#     paths.append(path)
# # get the data path
# cwd = os.getcwd()
# dataset_path = os.path.join(cwd, "images")


transformer = transforms.Compose([
    transforms.Resize((178, 178)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # (0-255) to (-1,1)
])
img_transformed = transformer(Image.open("./dataset/person1/image10.jpg"))
cv2.imshow(img_transformed)
labels=["Prachin", "Anya"]
with torch.no_grad():
    outputs = model(torch.reshape(img_transformed, (1,3,178,178)))
    predicted = torch.argmax(outputs, dim=1)
    print(outputs.data)
    print(labels[predicted[0]])
