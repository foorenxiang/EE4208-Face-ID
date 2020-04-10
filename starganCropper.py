'''
base code adapted from
https://towardsdatascience.com/a-guide-to-face-detection-in-python-3eab0f6b9fc1
https://medium.com/@sebastiannorena/pca-principal-components-analysis-applied-to-images-of-faces-d2fc2c083371
'''

import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
import glob
import os
import numpy as np

destDirectory = "./starganCropperOutput/"

if not os.path.isdir(destDirectory):
    try:
        os.mkdir(destDirectory)
    except:
        raise Exception("Unable to create directory!")

path = "./ganinput/"
fileExt = ".jpg"

if not os.path.isdir(path):
    raise Exception("Unable to find input directory!")

dataLoc = "/Users/foorx/opencv/venv/lib/python3.6/site-packages/cv2/data"

if not os.path.isdir(dataLoc):
    raise Exception("Unable to find cv2/data/!")

# cascPath = dataLoc + "/haarcascade_frontalface_default.xml"
cascPath = dataLoc + "/haarcascade_frontalface_alt.xml"
eyePath = dataLoc + "/haarcascade_eye.xml"
smilePath = dataLoc + "/haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyePath)
smileCascade = cv2.CascadeClassifier(smilePath)

files = list(map(lambda file: file.strip(), glob.glob(path+"*"+fileExt))) #strip white spaces from filename

expressionCount = 7
lastFileIndex = 0

#resize subimage
dim = (100, 100) #width, height

nameList = list()
expressionList = ("neutral", "eyes_closed", "sad", "smiling", "surprised", )

for file,index in zip(files,range(len(files))):
    gray = cv2.imread(file,0)
    
    # Detect faces in input image
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,flags=cv2.CASCADE_SCALE_IMAGE)
    print("Count of faces: " + str(len(faces)))

    peopleCount = len(faces) / expressionCount

    for face,faceIndex in zip(faces, range(len(faces))):
        personIndex = faceIndex / expressionCount + lastFileIndex # zero indexing
        personIndex = int(personIndex)
        expressionIndex = faceIndex % expressionCount # zero indexing
        expressionIndex = int(expressionIndex)

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
        cv2.imwrite(destDirectory + "person" + str(personIndex) + "_expr" + str(expressionIndex) + ".jpg",subImage)

        if faceIndex == len(faces)-1:
            lastFileIndex+=faceIndex