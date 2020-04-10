'''
base code adapted from
https://towardsdatascience.com/a-guide-to-face-detection-in-python-3eab0f6b9fc1
https://medium.com/@sebastiannorena/pca-principal-components-analysis-applied-to-images-of-faces-d2fc2c083371
'''

#use python3 command to execute!

import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils

# from sklearn.decomposition import PCA
from rxPCA import PCA
import numpy as np
import pandas as pd
from joblib import dump, load

font = cv2.FONT_HERSHEY_SIMPLEX

dataLoc = "/Users/foorx/opencv/venv/lib/python3.6/site-packages/cv2/data"

cascPath = dataLoc + "/haarcascade_frontalface_default.xml"
eyePath = dataLoc + "/haarcascade_eye.xml"
smilePath = dataLoc + "/haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyePath)
smileCascade = cv2.CascadeClassifier(smilePath)

import glob
import random
path = "./input/"
fileExt = ".jpg"
files = glob.glob(path+"*"+fileExt)
files = list(map(lambda file: file.strip(), files)) #strip white spaces from filename

couldNotDetects = list()
unstableDetects = list()
faceDict = dict() #stores subimages of detected faces
iVectorDict = dict() #stores vectors of faces in image space
pcDict = dict() #stores principal components of each face in eigenspace
fftDict = dict()

#use zero mean
useZeroMean = True

#resize subimage
dim = (60, 60) #width, height

#set to false to update models and dictionaries
reTrainAll = True
faceDictLoad = not reTrainAll
pcaModelLoad = not reTrainAll
pcaDataLoad = not reTrainAll
ELMLoad = False

processedFiles = list()

if faceDictLoad == False:
    for file in files:
        input = file
        # input = input.split('_') #convert to this instead
        # person = input[0][len(path):] #class
        # subclass = input[1].split('.')[0] #type of photo
        person = input[len(path):-len(fileExt)]
        print("Person:")
        print(person)
        gray = cv2.imread(file,0)
        
        # Detect faces in input image
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8,flags=cv2.CASCADE_SCALE_IMAGE)
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

            #vector of pixels for each face
            faceVector = subImage.flatten('C')

            if useZeroMean == True:
                #apply zero mean to remove brightness bias
                faceVector = faceVector-np.mean(faceVector)

            #add subImage to dictionary
            faceDict[person] = subImage
            #add facevector to dictionary
            iVectorDict[person] = faceVector
        else:
            print("Failed to detect face in image")
            couldNotDetects.append(person)

    print("Could not detects:")
    for person in couldNotDetects:
        print(person)
    print("Unstable detections:")
    for person in unstableDetects:
        print(person)
        #display images with unstable detections
        plt.figure(figsize=(8,8))
        plt.imshow(faceDict[person], cmap='gray')
        plt.show()      

    print("Number of faces in faceDict: " + str(len(faceDict)))
    dump(faceDict, 'faceDict.bin')
    dump(iVectorDict, 'iVectorDict.bin')

else:
    print("loaded processed subimages from disk")
    faceDict = load('faceDict.bin') #load subimages of detected faces
    iVectorDict = load('iVectorDict.bin') #load vectors of faces in image space

if pcaModelLoad == False:
    #determing principal components
    print("Performing PCA on dataset")
    pcaModel = PCA(n_components=.95)
    facesPDF = pd.DataFrame(list(iVectorDict.values()))
    print("number of detected faces")
    print(len(facesPDF))
    pcaModel.fit(facesPDF)
    dump(pcaModel, 'pcaModel.bin')
else:
    print("Loading PCA model from disk")
    pcaModel = load('pcaModel.bin')

if pcaDataLoad == False:
    print("Calcuating Principal Components of faces")
    for person in iVectorDict:
        pcDict[person] = pcaModel.transform(pd.DataFrame(list(iVectorDict[person])))
        print(person)
        print(pcDict[person])

    pcDict['useZeroMean'] = useZeroMean
    pcDict['dim'] = dim
    dump(pcDict, 'pcDict.bin')
    del pcDict['useZeroMean'] 
    del pcDict['dim'] 
    print("number of trained faces")
    print(len(pcDict))
else:
    print("Loading pcDict from disk")
    pcDict = load('pcDict.bin')

print("Keys of pcDict:")
print(pcDict.keys())

randPerson = random.choice(list(pcDict.keys()))
testFile = path+randPerson+".jpg"
testImg = cv2.imread(testFile,0)

faces = faceCascade.detectMultiScale(testImg, scaleFactor=1.1, minNeighbors=5,flags=cv2.CASCADE_SCALE_IMAGE)
face = None
#eliminate false positives
if len(faces): 
    face = faces[0]

    ########process subimage########
    x = face[0]
    y = face[1]
    w = face[2]
    h = face[3]
    #get subimage as square
    ext = min([max([w,h]), np.shape(testImg)[0]-x, np.shape(testImg)[1]]-y)
    # extract subimage containing face 
    subImage = testImg[y:(y+h),x:(x+w)]
    subImage = cv2.resize(subImage, dim)

    #vector of pixels for each face
    iVector = subImage.flatten('C')
    
    if useZeroMean == True:
        #apply zero mean to ignore brightnass bias
        iVector = iVector-np.mean(iVector)

    pcData = pcaModel.transform(pd.DataFrame(list(iVector)))

    #rewrite as map function
    guess = float('inf')
    currentError = float('inf')
    for person in pcDict:
        #get error between the PCA and perform SSD
        SSD = np.sum(np.subtract(pcData, pcDict[person]).flatten()**2)

        if SSD< currentError:
            currentError = SSD
            guess = person

    print("Person identified: " + guess)
    if guess == randPerson:
        print("Correct match!")
    else:
        print("Wrong identification. Correct person is: " + randPerson)