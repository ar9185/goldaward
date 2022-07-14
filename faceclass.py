import cv2
import numpy as mp
class face_detection:
	def __init__(self,imagesize,cascade_file):
		self.cascade_file = cascade_file
		self.imagesize=imagesize

		#gets Cascade
		self.faceCascade =cv2.CascadeClassifier(self.cascade_file)

	#reads the images and gives the image array 
	def readfile(self,imagePath):
		image=cv2.imread(imagePath,cv2.IMREAD_COLOR)
		return image
	#opens the image and read and scalefactor to ajust face where the face is in the image and minNeighbor shows the number of objects  next each other
	#minSize and gives the size of the window that the code will look at 
	def detect(self, image):
		coords=[]
		faces=self.faceCascade.detectMultiScale(
			image,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(100,100)
		)
		print(len(faces))
		number_faces=len((faces))
		if len(faces): print("Found {0} face!".format(len(faces)))
		for  (x,y,w,h) in faces:
			print("I am here")
			cv2.rectangle(image,(x+50,y+50),(x+w+50, y+h+50),(0,255,0),2)
			coords=[x,y,w,h]
		return image,faces,coords

	def live(self,video_capture):
		video_capture.set(3,640)
		video_capture.set(4,480)
		while True:
			ret, frame =video_capture.read()
			gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces= self.faceCascade.detectMultiScale(
				gray_img,
				scaleFactor=1.1,
				minNeighbors=5,
				minSize=(100,100)
			)
			for (x,y,w,h) in faces:
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.imshow("Face Found",frame)
			if  cv2.waitKey(1) & 0xFF==ord('q'):
				print("got here")
				break

		video_capture.release()
		cv2.destroyAllWindows()


