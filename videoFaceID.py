'''
base code adapted from
https://towardsdatascience.com/a-guide-to-face-detection-in-python-3eab0f6b9fc1
https://medium.com/@sebastiannorena/pca-principal-components-analysis-applied-to-images-of-faces-d2fc2c083371
'''

'''
Historical video face recognition script.

Prior to using this script, run trainFace.py to fit PCA model and classifier

use python3 command to execute!
'''

import cv2

import matplotlib.pyplot as plt
import dlib
from imutils import face_utils

# from sklearn.decomposition import PCA
from rxPCA import PCA
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
video_capture = cv2.VideoCapture('test1.mov')

#Fitted PCA model
print("Loading PCA model from disk")
pcaModel = load('pcaModel.bin')

#Dictionary of person and principal components
print("Loading pcDict from disk")
pcDict = load('pcDict.bin')

personClassifierModel = load('svcModel.bin')
expressionClassifierModel = load('svcExpressionsModel.bin')

#get from pcDict if useZeroMean is enabled
useZeroMean = False
#tune this parameter to speed up/slow down SSD #
#set to None to use all principal components
maxPrincipalComponents = None

if 'useZeroMean' in pcDict:
    useZeroMean = pcDict['useZeroMean']
    del pcDict['useZeroMean']

#subimage dimension
#must match dimensions used to train PCA
#get subImage dimensions from pcDict
try:
    dim = pcDict['dim']
    print("dimensions detected: " + str(dim))
    del pcDict['dim']
except KeyError:
    dim = (100, 100) #width, height

unknownPersonErrorThreshold = 650000

skipFrameThreshold = 2
skipFrame = skipFrameThreshold-1

validClassificationMethod = ['SSD', 'SVM']
classificationMethod = 'SSD'
if classificationMethod not in validClassificationMethod:
    raise ValueError("classificationMethod must be in " + str(validClassificationMethod) + '!')
print('classificationMethod: ' + classificationMethod)

while 1:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    

    if skipFrameThreshold <= 1 or  validClassificationMethod == 'SSD':
        pass
    else:
        skipFrame+=1
        if skipFrame == skipFrameThreshold:
            skipFrame = 0

        if skipFrame != 0:
            continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.7,
        minNeighbors=7,
        minSize=(450, 450),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    #for debugging
    subImages = list()
    faceIVectors = list() #list of face subImages in imagespace

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

        if classificationMethod == 'SSD':
            #apply SSD to guess person
            guess = float('inf')
            currentError = float('inf')
            for person in pcDict:
                #get error between the PCA and perform SSD
                SSD = np.sum(np.subtract(facePC, pcDict[person]).flatten()**2)

                if SSD< currentError:
                    currentError = SSD
                    guess = person

            if currentError>unknownPersonErrorThreshold:
                guess = "Unknown person"

            guess = guess.split('_')[0]
            print("Guessed person: " + guess)
            print("SSD Error: " + str(currentError))
            cv2.putText(frame, guess ,(x, y), font, 1,(0,255,0),5)

        if classificationMethod == 'SVM':
            #using SVM to classify person
            #must enlist sample before feeding to model
            personGuess = personClassifierModel.predict([facePC])
            #take first result from singleton list
            personGuess = personGuess[0]

            #must enlist sample before feeding to model
            expressionGuess = expressionClassifierModel.predict([facePC])
            #take first result from singleton list
            expressionGuess = expressionGuess[0]

            #using SSD to check if person is not in any class
            currentError = float('inf')
            for person in pcDict:
                if personGuess in person:
                #get error between the PCA and perform SSD
                    SSD = np.sum(np.subtract(facePC, pcDict[person]).flatten()**2)

                    if SSD<currentError:
                        currentError = SSD

            print('Squared Euclidean Distance: ' + str(currentError))
            if currentError>unknownPersonErrorThreshold:
                personGuess = "Unknown Person"

            print("Guessed person: " + personGuess)
            print("Guessed expression: " + expressionGuess)

            cv2.putText(frame, personGuess + '(' + expressionGuess +')' ,(x, y), font, 1,(255,0,0),5)

        cv2.imshow('subImage', subImage)

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