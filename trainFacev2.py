'''
base code adapted from
https://towardsdatascience.com/a-guide-to-face-detection-in-python-3eab0f6b9fc1
https://medium.com/@sebastiannorena/pca-principal-components-analysis-applied-to-images-of-faces-d2fc2c083371
'''

#!/usr/bin/env python3

'''
improvements:
use eye detector to confirm detected face is an actual face
increase depth of database dictionaries
''' 

import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils

from sklearn.decomposition import PCA
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
# path = "./input/caspeal/"
# files = glob.glob(path+"*.tif")
path = "./input/"
files = glob.glob(path+"*.jpg")
files = list(map(lambda file: file.strip(), files)) #strip white spaces from filename

couldNotDetects = list()
unstableDetects = list()
faceDict = dict() #stores subimages of detected faces
iVectorDict = dict() #stores vectors of faces in image space
pcDict = dict() #stores principal components of each face in eigenspace
fftDict = dict()

#use zero mean
useZeroMean = True

#resize subimage to 100x100
dim = (30, 30) #width, height

#set to false to update models and dictionaries
reTrainAll = True
faceDictLoad = not reTrainAll
pcaModelLoad = not reTrainAll
pcaDataLoad = not reTrainAll
ELMLoad = False

#override above
faceDictLoad = False

processedFiles = list()

if faceDictLoad == False:
    for file in files:
        input = file.split('_') #convert to this instead
        person = input[0][len(path):].lower() #class
        try:
            subclass = input[1].split('.') #type of photo
            subclass = subclass[0].lower()
        except:
            subclass = "Generic"
            print("************NO SUBCLASS SPECIFIED IN FILENAME************")
            #adjust person
            person = person.split('.')
            person = person[0]
        print("Person: " + person)
        print("Photo subclass: " + subclass)
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

            #vector of pixels for each face
            faceVector = subImage.flatten('C')

            if useZeroMean == True:
                #apply zero mean to remove brightness bias
                faceVector = faceVector-np.mean(faceVector)

            #add subImage to dictionary
            faceDict[person] = subImage
            # print("faceVector type: " + str(type(faceVector)))

            # faceDict data structure:
            # "person":{ "subclass1": [subImage 1, subImage 2], 
                      # "subclass2": [subImage 1,...
            #subImage type: numpy.ndarray
            #faceVector type: numpy.ndarray'

            #check if person key exists
            if not person in faceDict:
                # create new key for person
                faceDict[person] = {subclass: [subImage]}
            else:
                #check if subclass exists
                try:
                    #subclass exists
                    subclassInDB = faceDict[person][subclass]
                    #if subimage does not exist
                    if subclassInDB.index(subImage) == -1:
                        #add subimage to subclass
                        tempList = faceDict[person][subclass]
                        tempList.append(subImage)
                        faceDict[personData][subclass] = tempList
                    else:
                        #if subimage does not exist in subclass
                        pass
                except KeyError:
                    #subclass does not exist for person
                    #create subclass and insert new list with subImage
                    faceDict[person][subclass] = [subImage]
                except:
                    pass
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
    pcaModel = PCA(n_components=0.9)
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

    pcDict['useZeroMean'] = useZeroMean
    pcDict['dim'] = dim
    dump(pcDict, 'pcDict.bin')
    del pcDict['useZeroMean'] 
    print("number of trained faces")
    print(len(pcDict))
else:
    print("Loading pcDict from disk")
    pcDict = load('pcDict.bin')

if ELMLoad == False:
    pass

    #implement train test split when more inputs are available per face
    # trainPercentage = 0.7
    # trainingDataTrain = trainingDataPDF[:int(trainPercentage*len(trainingDataPDF))]
    # trainingDataTest = trainingDataPDF[int(trainPercentage*len(trainingDataPDF)):]
    # # trainX <-- training observations [# points, # features]
    # # trainy <-- training labels [# points]
    # # testX <-- test observations [# points, # features]
    # # testy <-- test labels [# points]

    # trainX = trainingDataTrain.copy()
    # trainX.drop(['GPSspeedkph'], axis=1, inplace = True)
    # ###APPLYING NORMALISATION TO DATASET AS REQUIRED BY ELM
    # trainX = StandardScaler().fit_transform(trainX)
    # trainy = trainingDataTrain["GPSspeedkph"]
    # trainy = trainy.astype('int')
    # # Index(['timeDeltaus', 'currentSampleHz', 'timeus', 'rcCommand0', 'rcCommand1',
    #        # 'rcCommand2', 'rcCommand3', 'vbatLatestV', 'gyroADC0', 'gyroADC1',
    #        # 'gyroADC2', 'accSmooth0', 'accSmooth1', 'accSmooth2', 'motor0',
    #        # 'motor1', 'motor2', 'motor3'],
    #       # dtype='object')

    # testX = trainingDataTest.copy()
    # testX.drop(['GPSspeedkph'], axis=1, inplace = True)
    # ###APPLYING NORMALISATION TO DATASET AS REQUIRED BY ELM
    # testX = StandardScaler().fit_transform(testX)
    # testy = trainingDataTest["GPSspeedkph"]

    # #########from plot_elm_comparison.p#########
    # kernel_names = ["tanh", "tribas", "hardlim", "rbf(0.1)"]
    # model_names = list(map(lambda name: name+"GPSSpeedModel", kernel_names))
    # nh = 10

    # # pass user defined transfer func
    # sinsq = (lambda x: np.power(np.sin(x), 2.0))
    # srhl_sinsq = SimpleRandomHiddenLayer(n_hidden=nh,
    #                                      activation_func=sinsq,
    #                                      random_state=0)

    # # use internal transfer funcs
    # srhl_tanh = SimpleRandomHiddenLayer(n_hidden=nh,
    #                                     activation_func='tanh',
    #                                     random_state=0)

    # srhl_tribas = SimpleRandomHiddenLayer(n_hidden=nh,
    #                                       activation_func='tribas',
    #                                       random_state=0)

    # srhl_hardlim = SimpleRandomHiddenLayer(n_hidden=nh,
    #                                        activation_func='hardlim',
    #                                        random_state=0)

    # # use gaussian RBF
    # srhl_rbf = RBFRandomHiddenLayer(n_hidden=nh*2, gamma=0.1, random_state=0)

    # log_reg = LogisticRegression(solver='liblinear')

    # classifiers = [ELMClassifier(srhl_tanh), ELMClassifier(srhl_tribas),ELMClassifier(srhl_hardlim),ELMClassifier(srhl_rbf)]
    # #########from plot_elm_comparison.p#########

    # #########only fit rbf kernel#########

    # model = ELMClassifier(srhl_rbf)
    # model.fit(trainX, trainy)

print("Keys of pcDict:")
print(pcDict.keys())

#test PCA dictionary
# randPerson = 'Jack'
# testFile = path+'jack1'+".png"
randPerson = random.choice(list(pcDict.keys()))
testFile = path+randPerson+".jpg"
testImg = cv2.imread(testFile,0)
# plt.figure(figsize=(8,8))
# plt.imshow(testImg, cmap='gray')
# plt.show()

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

# print("Number of faces loaded: " + str(len(faceDict)))
# print("Number of faces processed: " + str(len(pcDict)))