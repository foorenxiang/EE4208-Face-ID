'''
base code adapted from
https://towardsdatascience.com/a-guide-to-face-detection-in-python-3eab0f6b9fc1
https://medium.com/@sebastiannorena/pca-principal-components-analysis-applied-to-images-of-faces-d2fc2c083371
'''

'''
PCA model and classifiers training script

Run this script to train the PCA model, person and expression classifiers

This training script will generate and save the following files to disk:
faceDict.bin: subimages of input faces, as 2D Numpy arrays
iVectorDict.bin: flattened subimages of input faces, as 1D Numpy arrays
facesPDF.bin: flattened subimages of input faces, as Pandas Dataframe
pcaModel.bin: fitted PCA model
pcDict.bin: eigencoordinates of all faces processed with PCA, with additional keys input photo dimensions, zero mean setting (lighting compensation) and eigenspace centroid of all faces, as Python dictionary
svcModel.bin: trained SVM person classifier
adaBoostModel.bin: trained Adaboost person classifier
gradBoostModel.bin: trained gradient boosting person classifier
XGBoostModel.bin: trained XGBoost person classifier
randomForestModel.bin: trained random forest person classifier
stackingModel.bin: trained stacking generalizer person classifier
SSDModel.bin: trained Nearest Neighbour person classifier
3NNModel.bin: trained Nearest Neighbour person classifier, K = 3
5NNModel.bin: trained Nearest Neighbour person classifier, K = 5
ELMMLPModel.bin: trained ELM (MLP Hidden Layer) person classifier
ELMRBFModel.bin: trained ELM (RBF Hidden Layer) person classifier
expressionSSDstore.bin: nested list of expressions, containing list of SSD of eigencoordinates to centroid of that particular expressions
AdaboostExpressionsModel.bin: trained Adaboost expression classifier
svcExpressionsModel.bin: trained SVM expression classifier
rfExpressionsModel.bin: trained SVM expression classifier
centroidKNNExpressionsModel.bin: trained centroid based K-NN  expression classifier
SSDExpressionsModel.bin: Nearest Neighbour classifier
7NNExpressionsModel.bin: K-NN expression classifier, K = 7
10NNExpressionsModel.bin: K-NN expression classifier, K = 10
15NNExpressionsModel.bin: K-NN expression classifier, K = 15
20NNExpressionsModel.bin: K-NN expression classifier, K = 20
20NNExpressionsModel.bin: K-NN expression classifier, K = 20
XGBoostExpression.bin: trained XGBoost expression classifer


Script variables used for settings: 
    reTrainAll: set to true to
        Run face cropping from input folder
        Fit PCA model
        Project all faces in input photos to eigenspace
    faceDictLoad: load cropped face images instead of scanning input photos folder
    pcaModelLoad: load pre-fitted PCA model. PCA expected dimensions will be loaded as well. If photos were zero-meaned, the setting will be loaded as well.
    pcaDataLoad: load eigencoordinates of faces in training set
    disableTrainTestSplit: all input photos will be used to train classifiers. Enable this mode when training classifiers to be deployed for real-time/historical video detection. Disable this mode if evaluating classifier performance using input photos
    showIndividualPredictions: display each individual prediction by classifiers for debugging
    displayUnstableDetects: display flagged subimages if mulitple faces are detected in an input photo

Person and expression classifiers are always retrained when this script is run.

To exclude certain people for person classifiers, make use of script 'adjustDict.py'.
Indicate which people are to be removed in that script.
Train person classifiers on the reduced person set by placing the adjusted pcDict.bin in the same directory while setting reTrainAll parameter in this script to False 

use python3 command to execute!
'''

import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils

from rxPCA import PCA
import numpy as np
import pandas as pd
from joblib import dump, load
import time

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

# folder where training images are located
path = "./input/"
# extension of training images
fileExt = ".jpg"
files = glob.glob(path+"*"+fileExt)
files = list(map(lambda file: file.strip(), files)) #strip white spaces from filename

couldNotDetects = list()
unstableDetects = list()
faceDict = dict() #stores subimages of detected faces
iVectorDict = dict() #stores vectors of faces in image space
pcDict = dict() #stores principal components of each face in eigenspace

#use zero mean
useZeroMean = True

#resize subimage
dim = (100, 100) #width, height

#set to false to update models and dictionaries
reTrainAll = True
faceDictLoad = not reTrainAll
pcaModelLoad = not reTrainAll
pcaDataLoad = not reTrainAll

disableTrainTestSplit = False
showIndividualPredictions = False
displayUnstableDetects = False

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
        # display images with unstable detections
        if displayUnstableDetects:
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
    pcaModel = PCA(n_components=0.95)
    facesPDF = pd.DataFrame(list(iVectorDict.values()))
    dump(facesPDF, 'facesPDF.bin')
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
    t = time.time()
    for person, personIndex in zip(iVectorDict, range(len(iVectorDict))):
        print("Image Index:")
        print(personIndex)
        pcDict[person] = pcaModel.transform(pd.DataFrame(list(iVectorDict[person])))
        print(person)
        print(pcDict[person])
    elapsed = time.time() - t
    print("Dataset projection time: " + str(elapsed) + 'seconds')
    print(elapsed)
    ignoreKeys = ['useZeroMean', 'dim', 'facesCentroid']
    try:
        del accumulator
    except:
        pass
    for key,value in pcDict.items():
        if key not in ignoreKeys:
            if 'accumulator' not in globals():
                accumulator = value
            else:
                accumulator = np.add(accumulator,value)
    facesCentroid = accumulator.tolist()
    pcDict['facesCentroid'] = facesCentroid
    pcDict['useZeroMean'] = useZeroMean
    pcDict['dim'] = dim
    dump(pcDict, 'pcDict.bin')
    del pcDict['facesCentroid']
    del pcDict['useZeroMean'] 
    del pcDict['dim'] 
    print("number of trained faces")
    print(len(pcDict))
else:
    print("Loading pcDict from disk")
    pcDict = load('pcDict.bin')
    facesCentroid = pcDict['facesCentroid']
    del pcDict['facesCentroid']
    del pcDict['useZeroMean']
    del pcDict['dim']

print("Number of photos: " + str(len(pcDict)))
persons = set(map(lambda label: label.split('_')[0],pcDict.keys()))

print("Number of people: " + str(len(persons)))
print(persons)

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

print("Breakdown of person labels in training set (before train test split)")
personLabelCount = {}
for key in set(y):
    personLabelCount[key] = 0
for label in y:
    personLabelCount[label]+=1
print(personLabelCount)

print("Breakdown of expression labels in training set (before train test split)")
expressionLabelCount = {}
for key in set(expressionsy):
    expressionLabelCount[key] = 0
for label in expressionsy:
    expressionLabelCount[label]+=1
print(expressionLabelCount)

from sklearn.model_selection import train_test_split
test_size=0.2

#split for persons classes
trainX, testX, trainy, testy = train_test_split(X, y, test_size=test_size)

if disableTrainTestSplit:
    trainX, trainy = X, y
    
#split for expressions classes
expressionsTrainX, expressionsTestX, expressionsTrainy, expressionsTesty = train_test_split(expressionsX, expressionsy, test_size=test_size)

if disableTrainTestSplit:
    expressionsTrainX, expressionsTrainy = expressionsX, expressionsy

if disableTrainTestSplit:
    print("TRAIN TEST SPLIT DISABLED")
else:
    print("Train test split: ")
    print("Test size = " + str(test_size*100) + "%")

print("Breakdown of person labels in training set")
personLabelCount = {}
for key in set(trainy):
    personLabelCount[key] = 0
for label in trainy:
    personLabelCount[label]+=1
print(personLabelCount)

print("Breakdown of person labels in test set")
personLabelCount = {}
for key in set(testy):
    personLabelCount[key] = 0
for label in testy:
    personLabelCount[label]+=1
print(personLabelCount)

print("Breakdown of expression labels in training set")
personLabelCount = {}
for key in set(expressionsTrainy):
    personLabelCount[key] = 0
for label in expressionsTrainy:
    personLabelCount[label]+=1
print(personLabelCount)

print("Breakdown of expression labels in test set")
personLabelCount = {}
for key in set(expressionsTesty):
    personLabelCount[key] = 0
for label in expressionsTesty:
    personLabelCount[label]+=1
print(personLabelCount)

#####SVM#####
from sklearn.svm import SVC as classifier
model = classifier(kernel='linear',C=0.025)
model.fit(trainX,trainy)
dump(model, 'svcModel.bin')

print("##########Testing person identification with SVM model##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
            print(prediction + ' vs ' + actual)
            print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####Adaboost#####
from sklearn.ensemble import AdaBoostClassifier as classifier
from sklearn.tree import DecisionTreeClassifier
model = classifier(base_estimator = DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1, random_state=0)
model.fit(trainX,trainy)
dump(model, 'adaBoostModel.bin')


print("##########Testing person identification with Adaboost model##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
            print(prediction + ' vs ' + actual)
            print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####GradientBoosting#####
from sklearn.ensemble import GradientBoostingClassifier as classifier
model = classifier()
model.fit(trainX,trainy)
dump(model, 'gradBoostModel.bin')


print("##########Testing person identification with GradientBoosting model##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
            print(prediction + ' vs ' + actual)
            print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####XGBoost#####
from xgboost import XGBClassifier as classifier
model = classifier()
model.fit(np.array(trainX),np.array(trainy))
dump(model, 'XGBoostModel.bin')


print("##########Testing person identification with XGBoost model##########")
predictions = model.predict(np.array(testX))

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
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


print("##########Testing person identification with Random Forest model##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
            print(prediction + ' vs ' + actual)
            print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####Stacking Classifier#####
from sklearn.ensemble import StackingClassifier as classifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
estimators = [('rf', RandomForestClassifier()),('svc', SVC(kernel='linear',C=0.025))]
model = classifier(estimators = estimators, final_estimator=SVC(kernel='linear',C=0.025))

model.fit(trainX,trainy)
dump(model, 'stackingModel.bin')

print("##########Testing person identification with Stacking Classifier model with log reg as final estimator##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
            print(prediction + ' vs ' + actual)
            print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")


'''
#####HistGradientBoostingClassifier#####
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier as classifier
model = classifier()
model.fit(trainX,trainy)
dump(model, 'histGradBoostingModel.bin')

print("##########Testing person identification with HistGradientBoostingClassifier model##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
            print(prediction + ' vs ' + actual)
            print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")
'''

#####K-NN, n=1#####
from sklearn.neighbors import KNeighborsClassifier as classifier
model = classifier(n_neighbors=1)
model.fit(trainX,trainy)
dump(model, 'SSDModel.bin')

print("##########Testing person identification with K-NN model, K = 1 (SSD)##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
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
dump(model, '5NNModel.bin')

print("##########Testing person identification with K-NN model, K = 5##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
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
dump(model, '3NNModel.bin')

print("##########Testing person identification with K-NN model, K = 3##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
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
dump(model, 'ELMMLPModel.bin')

print("##########Testing person identification with ELM with MLP Random Layer model##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
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
dump(model, 'ELMRBFModel.bin')

print("##########Testing person identification with ELM with RBF Random Layer model##########")
predictions = model.predict(testX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, testy):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
            print(prediction + ' vs ' + actual)
            print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####calculate SSD for each expression#####
expressionSSDstore = {}
for expression in expressions:
    expressionSSDstore[expression] = []
for coordinate, label in zip(expressionsX, expressionsy):
    SSD = np.sum(np.subtract(facesCentroid, coordinate).flatten()**2)
    print(label + " " +str(SSD))
    expressionSSDstore[label].append(SSD)
from statistics import mean, median
for expression in expressions:
    print(expression)
    print(expressionSSDstore[expression])
    print("median:")
    print(median(expressionSSDstore[expression]))
    print("mean:")
    print(mean(expressionSSDstore[expression]))
    print("\n")

dump(expressionSSDstore, 'expressionSSDstore.bin')

#####Adaboost for expressions#####
from sklearn.ensemble import AdaBoostClassifier as classifier
from sklearn.tree import DecisionTreeClassifier
model = classifier(base_estimator = DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1, random_state=0)
model.fit(expressionsTrainX,expressionsTrainy)
dump(model, 'AdaboostExpressionsModel.bin')


print("##########Testing Expressions Adaboost model##########")
predictions = model.predict(expressionsTestX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, expressionsTesty):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
            print(prediction + ' vs ' + actual)
            print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

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
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
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
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
            print(prediction + ' vs ' + actual)
            print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####expression K-NN centroid#####
from sklearn.neighbors import NearestCentroid as classifier
model = classifier()
model.fit(expressionsTrainX,expressionsTrainy)
dump(model, 'centroidKNNExpressionsModel.bin')


print("##########Testing expression K-NN centroid model##########")
predictions = model.predict(expressionsTestX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, expressionsTesty):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
            print(prediction + ' vs ' + actual)
            print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####expression SSD#####
from sklearn.neighbors import KNeighborsClassifier as classifier
model = classifier(n_neighbors=1)
model.fit(expressionsTrainX,expressionsTrainy)
dump(model, 'SSDExpressionsModel.bin')


print("##########Testing expression SSD model##########")
predictions = model.predict(expressionsTestX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, expressionsTesty):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
            print(prediction + ' vs ' + actual)
            print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####expression K-NN, n=10#####
from sklearn.neighbors import KNeighborsClassifier as classifier
model = classifier(n_neighbors=10)
model.fit(expressionsTrainX,expressionsTrainy)
dump(model, '10NNExpressionsModel.bin')


print("##########Testing expression K-NN model, K = 10##########")
predictions = model.predict(expressionsTestX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, expressionsTesty):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
            print(prediction + ' vs ' + actual)
            print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####expression K-NN, n=20#####
from sklearn.neighbors import KNeighborsClassifier as classifier
model = classifier(n_neighbors=20)
model.fit(expressionsTrainX,expressionsTrainy)
dump(model, '20NNExpressionsModel.bin')


print("##########Testing expression K-NN model, K = 20##########")
predictions = model.predict(expressionsTestX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, expressionsTesty):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
            print(prediction + ' vs ' + actual)
            print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####expression K-NN, n=15#####
from sklearn.neighbors import KNeighborsClassifier as classifier
model = classifier(n_neighbors=15)
model.fit(expressionsTrainX,expressionsTrainy)
dump(model, '15NNExpressionsModel.bin')


print("##########Testing expression K-NN model, K = 15##########")
predictions = model.predict(expressionsTestX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, expressionsTesty):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
            print(prediction + ' vs ' + actual)
            print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####expression K-NN, n=7#####
from sklearn.neighbors import KNeighborsClassifier as classifier
model = classifier(n_neighbors=7)
model.fit(expressionsTrainX,expressionsTrainy)
dump(model, '7NNExpressionsModel.bin')


print("##########Testing expression K-NN model, K = 7##########")
predictions = model.predict(expressionsTestX)

rights, wrongs = 0,0
for prediction, actual in zip(predictions, expressionsTesty):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
            print(prediction + ' vs ' + actual)
            print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")

#####XGBoost#####
from xgboost import XGBClassifier as classifier
model = classifier()
model.fit(np.array(expressionsTrainX),np.array(expressionsTrainy))
dump(model, 'XGBoostExpression.bin')


print("##########Testing expression XGBoost model##########")
predictions = model.predict(np.array(expressionsTestX))

rights, wrongs = 0,0
for prediction, actual in zip(predictions, expressionsTesty):
    if prediction == actual:
        if showIndividualPredictions:
            print(prediction)
            print("Correct!")
        rights+=1
    else:
        if showIndividualPredictions:
            print(prediction + ' vs ' + actual)
            print("Wrong!")
        wrongs+=1

print("Rights: " + str(rights))
print("Wrongs:" + str(wrongs))
print("Score: " + str(rights/(rights+wrongs)*100) + "%")
print("\n")
