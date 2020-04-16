#use python3 command to execute!

# from sklearn.decomposition import PCA
from rxPCA import PCA
import numpy as np
import pandas as pd
from joblib import dump, load


pcDict = load('pcDict40_40.bin') 
del pcDict['useZeroMean']
del pcDict['dim']

######define classes######
#extract person name from each key using map
#cast to set to make selection unique
persons = set(map(lambda label: label.split('_')[0],pcDict.keys()))
print(persons)

#for each person, create an svm model to determine if it is that person
for person in persons:
    #generate dataset for training svm model
    for fileName, eigenCoordinates in pcDict.items():

        if person in fileName:

        else: