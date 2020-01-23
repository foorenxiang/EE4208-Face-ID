'''
base code adapted from
https://towardsdatascience.com/a-guide-to-face-detection-in-python-3eab0f6b9fc1
https://medium.com/@sebastiannorena/pca-principal-components-analysis-applied-to-images-of-faces-d2fc2c083371
'''

import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils

import sklearn.decomposition as pca
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

dataLoc = "/Users/foorx/opencv/venv/lib/python3.6/site-packages/cv2/data"

cascPath = dataLoc + "/haarcascade_frontalface_default.xml"
eyePath = dataLoc + "/haarcascade_eye.xml"
smilePath = dataLoc + "/haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyePath)
smileCascade = cv2.CascadeClassifier(smilePath)


##############STATIC FACE DETECTION##############
import glob
import random
# path = "./input/caspeal/"
# files = glob.glob(path+"*.tif")
path = "./input/"
files = glob.glob(path+"*.jpg")
input = random.choice(files)
print("Person:")
print(input[len(path):-4])
gray = cv2.imread(input,0)
colour = cv2.imread(input)
b,g,r = cv2.split(colour)
colour = cv2.merge([r,g,b])

# Detect faces
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,flags=cv2.CASCADE_SCALE_IMAGE)
# For each face
for (x, y, w, h) in faces: 
    # Draw rectangle around the face
    cv2.rectangle(colour, (x, y), (x+w, y+h), (255, 255, 255), 3)

plt.figure(figsize=(12,8))
plt.imshow(gray, cmap='gray')
plt.imshow(colour)
plt.show()
##############STATIC FACE DETECTION##############


##############PCA##############
faceVectors = list() #list containing faceVectors
faceImages = list()

pcaKernel = pca.KernelPCA(n_components=None, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None, alpha=1.0, fit_inverse_transform=False, eigen_solver='auto', tol=0, max_iter=None, remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=None)
    
if len(faces):
    for (x, y, w, h) in faces:
        #get subimage as square
        ext = min([max([w,h]), np.shape(gray)[0]-x, np.shape(gray)[1]]-y)
        # extract subimage containing face 
        subImage = gray[y:(y+h),x:(x+w)]
        #resize subimage to 100x100
        dim = (100, 100) #width, height
        subImage = cv2.resize(subImage, dim)
        faceImages.append(subImage)

        #vector of pixels for each face
        faceVector = subImage.flatten('C')
        print(type(faceVector))
        print(np.shape(faceVector))
        faceVectors.append(faceVector)

        plt.figure(figsize=(8,8))
        plt.imshow(subImage, cmap='gray')      
        plt.show()

