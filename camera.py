
import cv2 
from faceclass import face_detection
import os

class Camera:
	"""	
	camera class that will take care of  camera functions using opencv
	camera will be on video mode until ended
	build abilities to have object detection
	"""
	
	def __init__(self, camera_index = 0, imagesize = (640, 640)):
		self.camera_index = camera_index #where the camera is
		try:
			self.video_capture = cv2.VideoCapture(self.camera_index) #the camera object itself
		except:
			print(camera_index, "is not a valid index")
		self.imagesize = imagesize  #size of the image you would like

		self.video_capture.set(3,imagesize[0]) #width and height provided
		self.video_capture.set(4,imagesize[1])


		self.faceDetector = None #face detector is the user would like
		self.out = None #if the user would like ot save the video
	
	def start(self, face_detect = False, save = False, cascade_file = None):
		""" start the camera and start reading video input
			set face_detect to true if you want the camera to face detect
			set save to true if you would like to save the video
			goal: add object detection  """
		if save:
			fourcc = cv2.VideoWriter_fourcc(*'XVID')
			self.out = cv2. VideoWriter('output.avi',20,0,(self.imagesize))
						
		if face_detect:
			if not cascade_file: 
				print("no cascade file was provided")
				return
			self.faceDetector = face_detection(self.imagesize, cascade_file)
	

	def run(self, person_num, num, face_detect=False):
		#will keep the camera on until user shuts it off
		#will need to be able to handle error and cut it when the server shuts off 
		ret, frame = self.video_capture.read()
		img=frame.copy()
		stop = False
		print(ret, face_detect)
		frame2, faces,points= self.faceDetector.detect(frame)
		cv2.imshow("Image",img)
		if ret and len(points)!=0 and face_detect==True:
			print(points)
			roi_img=img[points[1]:points[1]+points[3], points[0]:points[0]+points[2]]
			dirname = "dataset/person"+str(person_num)
			if not os.path.exists(dirname):
				os.mkdir(dirname)
			cv2.imwrite("./dataset/person"+str(person_num)+"/image"+str(num)+".jpg",roi_img)

		'''
		if ret:

			#find any faces in the fr			if face_detect: frame,faces,points = self.faceDetector.detect(frame)


			#roi_img= img[points[1]:points[1]+points[3], points[0]:points[0]+points[2]]

			#save the image to our output video
			if save:self.out.write(img)


			cv2.imshow("Image",img)

		'''
		#end the camera now if 'e' is pressed
		if cv2.waitKey(1) & 0xFF == ord('e'):
			self.stop()
			stop = True
			return img, stop
		return None, stop 

	def faceDetect(self,image):
		""" detects the faces in the given image""" 
		return self.faceDetector.detect(image)
	'''
	def takePicture(self,imagenum):
		""" will take a single picture and save it"""
		ret, frame = self.video_capture.read()
		copyimg=frame.copy()
		if ret and cv2.waitKey(1) and 0xFF == ord('t'):
			test=cv2.imwrite("./dataset/images"+str(imagenum)+".jpg",frame)
			roi_img=copyimg[pointsreturn test
		else:
			print("unable to start camera")
	'''
	def stop(self):
		""" stop the camera """
		self.video_capture.release()
		if self.out:
			self.out.release()
		cv2.destroyAllWindows()	

