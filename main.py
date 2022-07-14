import cv2
import time
from faceclass import face_detection
from camera import Camera
import os
import glob
from torch_model import ImageReader, CNN
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision
from torch.optim import Adam, SGD
from torch.nn import Linear, ReLU, Sequential, MaxPool2d, Conv2d, Module, Softmax, CrossEntropyLoss


cam=Camera(camera_index= 0)
cascade_file="haarcascade_frontalface_default.xml"
cam.start(face_detect=True,cascade_file=cascade_file)
print("start")
#faces, stop =cam.run(face_detect=True)
i=0
person = 0
'''
names = []
while True:
	personId = input("Start a new  person?")
	personName= input("type the person's name")
	if personId == "Yes":
		names.append(personName)
		while i<300:
			i=i+1
			faces, stop = cam.run(person, i, face_detect=True)
#			print("got here")
			#print("taking pic  . . .")
			#pic=cam.takePicture(i)
			if stop:
				person += 1
				break
	else:
		break
'''

#get the data path
cwd = os.getcwd()
dataset_path = os.path.join(cwd,"dataset") 

#transform the data
transformer = transforms.Compose([
	transforms.Resize((128,128)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

#load the data with the above transformation
train_loader = DataLoader(torchvision.datasets.ImageFolder(dataset_path, transform = transformer),batch_size = 16,shuffle = True)

# reader = ImageReader(dataset_path)
# images = reader.mapImagesToTensor()
# #images = 

#create the model
model = CNN()
#output = model(images)

#optimizer and loss functions
optimizer = Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001 )
loss_function = CrossEntropyLoss()

num_epochs = 10
train_count = len(glob.glob(dataset_path + "/**/*.jpg" ))

#train model
for epoch in range(num_epochs):

	train_accuracy = 0.0
	model.train()
	for i, (images,labels) in enumerate(train_loader):

		optimizer.zero_grad()
		outputs = model(images)
		#calculate the loss and back propogate
		loss = loss_function(output,labels)
		loss.backward()
		optimizer.step()

		_,prediction = torch.max(outputs.data,1)
		train_accuracy += int(torch.sum(prediction=labels.data))

	train_accuracy = train_accuracy/train_count
	print("EPOCH: " + str(epoch) + "accuracy:  ", train_accuracy)

	model.eval()



#test model
# faces, stop = cam.run(0, 1000, face_detect=True)

# test_image = os.path.join(cwd,"dataset/person0/image1000.jpg")
# test_image = reader.loadAndReSizeImage(test_image)

# faceId = model(test_image)
# print(faceId)



