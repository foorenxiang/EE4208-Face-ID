import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
import glob
import numpy as np
import os

dataLoc = "/Users/foorx/opencv/venv/lib/python3.6/site-packages/cv2/data"

cascPath = dataLoc + "/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

path = "./"
fileExt = ".tif"
files = glob.glob(path+"*"+fileExt)

faceDict = dict() #stores subimages of detected faces
processedFiles = []

#resize subimage
dim = (256, 256) #width, height
destDir = './cropped_' + str(dim[0]) + '/'
try:
	os.mkdir(destDir)
except:
	pass

for file in files:
	input = file[:-len(fileExt)]
	person = input[0][len(path):]
	print("Person:")
	gray = cv2.imread(file,0)
   
	# Detect faces in input image
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,flags=cv2.CASCADE_SCALE_IMAGE)
	print("Count of faces: " + str(len(faces)))

	face = None
	#attempt to eliminate false positives, only take first face
	if len(faces): 
		processedFiles.append(person)
		if len(faces)>1:
			unstableDetects.append(person)

		face = faces[0]

		########process subimage########
		x = face[0]
		y = face[1]
		w = face[2]
		h = face[3]
		#get subimage as square
		ext = min([max([w,h]), np.shape(gray)[0]-x, np.shape(gray)[1]]-y)
		# extract subimage containing face 
		subImage = gray[y:(y+h),x:(x+w)]
		#resize subimage
		subImage = cv2.resize(subImage, dim)
	newFilePath = destDir + input + '.jpg'
	cv2.imwrite(newFilePath, subImage)