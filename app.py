from flask import *
from flask import request
import base64
import os
import torch
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
from network import CNN
from torch.utils.data import DataLoader
import torchvision
import cv2
import numpy as np


app = Flask('gold')

@app.route('/')
def main():
    return True

@app.route('/testing/post', methods=['POST'])
def home():
    if request.method=="POST":
        img_file = request.json["img_file"]
        file= base64.b64decode(img_file)
        print(img_file)
        with open("/Users/anyaranavat/PycharmProject/gold_award/images/testingimg.jpg", "wb") as l:
            saved_img=l.write(file)

        paths = []
        for pic in os.listdir("./images"):
            path = os.path.join("images", pic)
            paths.append(path)
        # get the data path
        cwd = os.getcwd()
        dataset_path = os.path.join(cwd, "images")

        print(paths)
        '''
        transformer = transforms.Compose([
            transforms.Resize((178, 178)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # (0-255) to (-1,1)
        ])

        img_transformed = transformer(Image.open(path))
        
        #imageFolder = torchvision.datasets.ImageFolder(path, transform=transformer)
        #
        # n = len(img_transformed)
        # n_test = int(0.25 * n)
        #print(img_transformed, range(n_test))
        # test_set = torch.utils.data.Subset(img_transformed, range(n_test))
        # test_loader = DataLoader(test_set, batch_size=16, shuffle=True)
        model = CNN()
        model.load_state_dict(torch.load("/Users/anyaranavat/PycharmProject/gold_award/best_checkout_model.pth"))
        
        labels=["Rakhi", "Anya"]
        with torch.no_grad():
            outputs = model(torch.reshape(img_transformed, (1,3,178,178)))
            predicted = torch.argmax(outputs, dim=1)
            print(outputs.data)
            return labels[predicted[0]]
        '''
        model = load_model('keras_model.h5')

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        # Replace this with the path to your image
        image = Image.open(paths[0])
        # resize the image to a 224x224 with the same strategy as in TM2:
        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        # turn the image into a numpy array
        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        labels = ['Rakhi', 'Anya', 'Nani']
        prediction = model.predict(data)
        return labels[np.argmax(prediction)]

        #_, prediction = torch.max(outputs.data, 1)
        #img_array = cv2.imread(path)

        #data2 = Image.fromarray(img_array)
        #data2.show()
        #data1=transformer(data2) #got [3,128,128] #expected[1,128,128]

        #print(data1)
        #print(data1.shape)

        # load the data with the above transformation



        #predicted_outputs = model(data1)

        #print('Accuracy of the model based on the test set of:', predicted_outputs)
        #model.eval()

        #prediction = model.predict([resized])
        #print(prediction)
        #return "success"

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)