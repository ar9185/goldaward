import cv2 
import numpy as np
from PIL import Image
import os


paths = []
faces = []
ids = []
for pic in os.listdir("./dataset"):
	path=os.path.join("dataset", pic)
	paths.append(path)

# print(paths)
for img_path in paths:
	image=Image.open(img_path).convert("L")

	imgNp =np.array(image, "uint8")
	faces.append(imgNp)
	id = img_path.split("/")[1].split(".")[0] #.split("e")[0]
	
	ids.append(id)
ids = np.array(ids)


#	print(id)


trainer = cv2.face.LBPHFaceRecognizer_create()

trainer.train(faces, ids)

trainer.write("training.yml")


