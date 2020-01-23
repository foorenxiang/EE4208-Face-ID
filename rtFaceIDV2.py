'''
base code adapted from
https://towardsdatascience.com/a-guide-to-face-detection-in-python-3eab0f6b9fc1
https://medium.com/@sebastiannorena/pca-principal-components-analysis-applied-to-images-of-faces-d2fc2c083371
'''

'''
improvements:
use eye detector to confirm detected face is an actual face
increase depth of database dictionaries
allow small faces to be detected in live video stream
''' 

import cv2

import matplotlib.pyplot as plt
import dlib
from imutils import face_utils

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from joblib import load

dataLoc = "/Users/foorx/opencv/venv/lib/python3.6/site-packages/cv2/data"

cascPath = dataLoc + "/haarcascade_frontalface_default.xml"
eyePath = dataLoc + "/haarcascade_eye.xml"
smilePath = dataLoc + "/haarcascade_smile.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyePath)
smileCascade = cv2.CascadeClassifier(smilePath)

font = cv2.FONT_HERSHEY_SIMPLEX
video_capture = cv2.VideoCapture(0)

#Fitted PCA model
print("Loading PCA model from disk")
pcaModel = load('pcaModel.bin')

#Dictionary of person and principal components
print("Loading pcDict from disk")
pcDict = load('pcDict.bin')



#tune this parameter to speed up/slow down SSD #
#set to None to use all principal components
maxPrincipalComponents = None

#get from pcDict if useZeroMean is enabled
useZeroMean = False
if 'useZeroMean' in pcDict:
    useZeroMean = pcDict['useZeroMean']
    del pcDict['useZeroMean']

#subimage dimension
#must match dimensions used to train PCA
#get subImage dimensions from pcDict
try:
    dim = pcDict['dim']
    del pcDict['dim']
except KeyError:
    dim = (100, 100) #width, height

print("subImage dimensions: " + str(dim))
while 1:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    #for debugging
    subImages = list()
    faceIVectors = list() #list of face subImages in imagespace
    facePCs = list()

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        

        # smile = smileCascade.detectMultiScale(
        #     roi_gray,
        #     scaleFactor= 1.16,
        #     minNeighbors=35,
        #     minSize=(25, 25),
        #     flags=cv2.CASCADE_SCALE_IMAGE
        # )

        # for (sx, sy, sw, sh) in smile:
        #     cv2.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
        #     cv2.putText(frame,'Smile',(x + sx,y + sy), 1, 1, (0, 255, 0), 1)

        # eyes = eyeCascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #     cv2.putText(frame,'Eye',(x + ex,y + ey), 1, 1, (0, 255, 0), 1)
        


    #######face identification#######



    # if len(faces):
    #     for (x, y, w, h) in faces:
        #get subimage as square
        ext = min([max([w,h]), np.shape(gray)[0]-x, np.shape(gray)[1]]-y)
        # extract subimage containing face 
        subImage = gray[y:(y+h),x:(x+w)]
        #resize subimage to 100x100
        subImage = cv2.resize(subImage, dim)
        #for debugging
        subImages.append(subImage)

        #vector of pixels for each face
        faceIVector = subImage.flatten('C')
        
        if useZeroMean == True:
            #apply zero mean to ignore brightnass bias
            faceIVector = faceIVector-np.mean(faceIVector)

        #for debugging
        faceIVectors.append(faceIVector)
        #required
        facePC = pcaModel.transform(pd.DataFrame(list(faceIVector)))
        facePCs.append(facePC)

        #apply SSD to guess person
        guess = float('inf')
        currentError = float('inf')
        for person in pcDict:
            person = person.lower()
            #get error between the PCA and perform SSD
            SSD = np.sum(np.subtract(facePC, pcDict[person]).flatten()**2)
            SSD = np.sum(np.subtract(facePC[:,:maxPrincipalComponents], pcDict[person][:,:maxPrincipalComponents]).flatten()**2)

            if SSD< currentError:
                currentError = SSD
                guess = person

        print("Guessed person: " + guess)
        print("SSD Error: " + str(currentError))
        renXiangError = np.sum(np.subtract(facePC[:,:maxPrincipalComponents], pcDict['foo ren xiang'][:,:maxPrincipalComponents]).flatten()**2)
        print('Error from "foo ren xiang": ' + str(renXiangError))
        print("Delta between renXiangError and SSDError: " + str(renXiangError - currentError))
        cv2.putText(frame, guess ,(x, y), font, 1,(255,0,0),5)

        np.sum(np.subtract(facePC, pcDict[person]).flatten()**2)

    print("Number of faces detected:")
    print(len(faces))

    cv2.putText(frame,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)      

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()