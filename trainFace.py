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
dim = (100, 100) #width, height

#set to false to update models and dictionaries
reTrainAll = False
faceDictLoad = not reTrainAll
pcaModelLoad = not reTrainAll
pcaDataLoad = not reTrainAll

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
        # plt.figure(figsize=(8,8))
        # plt.imshow(faceDict[person], cmap='gray')
        # plt.show()      

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
    pcaModel = PCA(n_components=30)
    facesPDF = pd.DataFrame(list(iVectorDict.values()))
    print("number of detected faces")
    print(len(facesPDF))
    if (len(facesPDF) == 0):
        raise ValueError
    pcaModel.fit(facesPDF)
    dump(pcaModel, 'pcaModel.bin')
else:
    print("Loading PCA model from disk")
    pcaModel = load('pcaModel.bin')

if pcaDataLoad == False:
    print("Calcuating Principal Components of faces")
    for person, personIndex in zip(iVectorDict, range(len(iVectorDict))):
        print("Image Index:")
        print(personIndex)
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
    del pcDict['useZeroMean']
    del pcDict['dim']

print("Number of photos: " + str(len(pcDict)))
persons = set(map(lambda label: label.split('_')[0],pcDict.keys()))

print("Number of people: " + str(len(persons)))
print(persons)

from sklearn.model_selection import train_test_split
X, y, expressionsy = [],[],[]
for fileName, eigenCoordinates in pcDict.items():
    X.append(eigenCoordinates)
    y.append(fileName.split('_')[0])
    expressionsy.append(fileName.split('_')[1])

expressionsX, newExpressionsy = [],[]
for eigenCoordinates, expression in zip(X, expressionsy):
    if not expression.isnumeric():
        expressionsX.append(eigenCoordinates)
        newExpressionsy.append(expression)

expressionsy = newExpressionsy

expressions = set(expressionsy)
print("Number of expressions: " + str(len(expressions)))
print(expressions)

test_size=0.2

#split for persons classes
trainX, testX, trainy, testy = train_test_split(X, y, test_size=test_size)
    
#split for expressions classes

expressionsTrainX, expressionsTestX, expressionsTrainy, expressionsTesty = train_test_split(expressionsX, expressionsy, test_size=test_size)

print("Train test split: ")
print("Test size = " + str(test_size*100) + "%")

#####SVM#####
from sklearn.svm import SVC as classifier
model = classifier(kernel='linear',C=0.025)
model.fit(trainX,trainy)
dump(model, 'svcModel.bin')


print("##########Testing SVM model##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        # print(prediction)
        # print("Correct!")
        rights+=1
    else:
        print(prediction + ' vs ' + actual)
        print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

# #####Adaboost#####
# from sklearn.ensemble import AdaBoostClassifier as classifier
# model = classifier(n_estimators=100, random_state=0)
# model.fit(trainX,trainy)
# dump(model, 'adaBoostModel.bin')


# print("##########Testing Adaboost model##########")
# predictions = model.predict(testX)

# rights, wrongs = 0,0
# for prediction, actual in zip(predictions, testy):
#     if prediction == actual:
#         print(prediction)
#         print("Correct!")
#         rights+=1
#     else:
#         print(prediction + ' vs ' + actual)
#         print("Wrong!")
#         wrongs+=1

# print("Rights: " + str(rights))
# print("Wrongs:" + str(wrongs))
# print("Score: " + str(rights/(rights+wrongs)*100) + "%")
# print("\n")

#####GradientBoosting#####
from sklearn.ensemble import GradientBoostingClassifier as classifier
model = classifier()
model.fit(trainX,trainy)
dump(model, 'gradBoostModel.bin')


print("##########Testing GradientBoosting model##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        # print(prediction)
        # print("Correct!")
        rights+=1
    else:
        print(prediction + ' vs ' + actual)
        print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####Random Forest#####
from sklearn.ensemble import RandomForestClassifier as classifier
model = classifier()
model.fit(trainX,trainy)
dump(model, 'randomForestModel.bin')


print("##########Testing Random Forest model##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        # print(prediction)
        # print("Correct!")
        rights+=1
    else:
        print(prediction + ' vs ' + actual)
        print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

# #####Stacking Classifier#####
# from sklearn.ensemble import StackingClassifier as classifier
# from sklearn.ensemble import RandomForestClassifier
# # from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# estimators = [('rf', RandomForestClassifier()),('svc', SVC(kernel='linear',C=0.025))]
# model = classifier(estimators = estimators, final_estimator=SVC(kernel='linear',C=0.025))

# model.fit(trainX,trainy)
# dump(model, 'stackingModel.bin')


# print("##########Testing Stacking Classifier model with log reg as final estimator##########")
# predictions = model.predict(testX)

# rights, wrongs = 0,0
# for prediction, actual in zip(predictions, testy):
#     if prediction == actual:
#         print(prediction)
#         print("Correct!")
#         rights+=1
#     else:
#         print(prediction + ' vs ' + actual)
#         print("Wrong!")
#         wrongs+=1

# print("Rights: " + str(rights))
# print("Wrongs:" + str(wrongs))
# print("Score: " + str(rights/(rights+wrongs)*100) + "%")
# print("\n")

#####HistGradientBoostingClassifier#####
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier as classifier
model = classifier()
model.fit(trainX,trainy)
dump(model, 'histGradBoostingModel.bin')


print("##########Testing HistGradientBoostingClassifier model##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        # print(prediction)
        # print("Correct!")
        rights+=1
    else:
        print(prediction + ' vs ' + actual)
        print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####K-NN, n=1#####
from sklearn.neighbors import KNeighborsClassifier as classifier
model = classifier(n_neighbors=1)
model.fit(trainX,trainy)
dump(model, 'KNNModel.bin')


print("##########Testing K-NN model, n = 1##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        # print(prediction)
        # print("Correct!")
        rights+=1
    else:
        print(prediction + ' vs ' + actual)
        print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####K-NN, n=5#####
from sklearn.neighbors import KNeighborsClassifier as classifier
model = classifier(n_neighbors=5)
model.fit(trainX,trainy)
dump(model, 'KNNModel.bin')


print("##########Testing K-NN model, n = 5##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        # print(prediction)
        # print("Correct!")
        rights+=1
    else:
        print(prediction + ' vs ' + actual)
        print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####K-NN, n=3#####
from sklearn.neighbors import KNeighborsClassifier as classifier
model = classifier(n_neighbors=3)
model.fit(trainX,trainy)
dump(model, 'KNNModel.bin')


print("##########Testing K-NN model, n = 3##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        # print(prediction)
        # print("Correct!")
        rights+=1
    else:
        print(prediction + ' vs ' + actual)
        print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####ELM with MLP Random Layer#####
from random_layer import MLPRandomLayer
from elm import GenELMClassifier as classifier
def powtanh_xfer(activations, power=1.0):
    return pow(np.tanh(activations), power)
model = classifier(hidden_layer=MLPRandomLayer(n_hidden=100, activation_func=powtanh_xfer, activation_args={'power':3.0}))
model.fit(trainX,trainy)
dump(model, 'ELMModel.bin')


print("##########Testing ELM with MLP Random Layer model##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        # print(prediction)
        # print("Correct!")
        rights+=1
    else:
        print(prediction + ' vs ' + actual)
        print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####ELM with RBF Random Layer#####
from random_layer import RBFRandomLayer
from elm import GenELMClassifier as classifier
model = classifier(hidden_layer=RBFRandomLayer(n_hidden=100, random_state=0, rbf_width=0.01))
model.fit(trainX,trainy)
dump(model, 'ELMModel.bin')


print("##########Testing ELM with RBF Random Layer model##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        # print(prediction)
        # print("Correct!")
        rights+=1
    else:
        print(prediction + ' vs ' + actual)
        print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")




expressionsTrainX, expressionsTestX, expressionsTrainy, expressionsTesty

#####SVM for expressions#####
from sklearn.svm import SVC as classifier
model = classifier(kernel='linear',C=0.025)
model.fit(expressionsTrainX,expressionsTrainy)
dump(model, 'svcExpressionsModel.bin')


print("##########Testing Expressions SVM model##########")
predictions = model.predict(expressionsTestX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, expressionsTesty):
    if prediction == actual:
        # print(prediction)
        # print("Correct!")
        rights+=1
    else:
        print(prediction + ' vs ' + actual)
        print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####Random Forest for expressions#####
from sklearn.ensemble import RandomForestClassifier as classifier
model = classifier()
model.fit(expressionsTrainX,expressionsTrainy)
dump(model, 'rfExpressionsModel.bin')


print("##########Testing Expressions Random Forest model##########")
predictions = model.predict(expressionsTestX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, expressionsTesty):
    if prediction == actual:
        # print(prediction)
        # print("Correct!")
        rights+=1
    else:
        print(prediction + ' vs ' + actual)
        print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")