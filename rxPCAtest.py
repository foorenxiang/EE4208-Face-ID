import cv2
import matplotlib.pyplot as plt
# import dlib
# from imutils import face_utils
import numpy as np
import pandas as pd
from joblib import load, dump
import glob
import random
from scipy import linalg
from numpy.linalg import eig
from functools import reduce

import pprint
pp = pprint.PrettyPrinter(indent=4)

import rxPCA

# #dataset must be numpy array
# def fit(dataset):
# 	#find mean of each feature
# 	#deduct mean of each feature from dataset
# 	datasetMean = np.mean(dataset, axis=0)
# 	dataset = dataset - datasetMean

# 	#find covariance matrix
# 	covMatrix = dataset.cov()

# 	#calculate eigen decomposition of covariance matrix (sort eigenvalues in descending order)
# 	# covMatrixDet = determinants(covMatrix)
# 	eigValues, eigVectors = eig(covMatrix)
# 	absEigValues = list(map(lambda x: abs(x), eigValues))
# 	principalComponents = []
# 	for eigValue, absEigValue, eigVector in sorted(zip(eigValues, absEigValues, eigVectors), key=lambda x: x[1], reverse = True):
# 		principalComponents.append({
# 			'eigValue': eigValue,
# 			'absEigValue': absEigValue,
# 			'eigVector': eigVector
# 			})

# 	eigenValuesSum = sum(absEigValues)

# 	varianceExplanation = 0.7
	
# 	varianceAggregator = 0
# 	componentsToKeep=0
# 	while (varianceAggregator/eigenValuesSum) < varianceExplanation:
# 		varianceAggregator+=principalComponents[componentsToKeep]['absEigValue']
# 		componentsToKeep+=1

# 	# print('Components to keep: ' + str(componentsToKeep))
# 	reducedPrincipalComponents = principalComponents[:componentsToKeep]

# 	# print("reducedPrincipalComponent: ")
# 	# pp.pprint(reducedPrincipalComponents)

# 	return reducedPrincipalComponents, datasetMean

# def transform(iVectorsTestPDF, reducedPrincipalComponents, datasetMean):
# 	#normalise test data
# 	iVectorsTestPDF -= datasetMean
# 	#project test data to eigenspace
# 		#iterate through each dataframe row
# 	projValues = []
# 	for index, row in iVectorsTestPDF.iterrows():
# 		projValue = []
# 		for component in reducedPrincipalComponents:
# 			projValueDim = component['eigVector'].dot(row)
# 			projValue.append(projValueDim)
# 		projValues.append(projValue)

# 	print("\n\nprojValues: ")
# 	pp.pprint(projValues)
# 	return projValues

def main():
	#load faceDict
	try:
		faceDict  = load('faceDict.bin')
	except:
		print("Unable to load faceDict from disk")
	iVectorDict = dict()
	#select k number of images to form dataset
	k = 10
	persons = random.choices(list(faceDict.keys()), k=k)
	for person in persons:
		try:
			subImage = faceDict[person]
		except:
			print("Unable to find image for " + person)

		iVectorDict[person] = subImage.flatten('C')
	print("iVectorDict keys: " + str(list(iVectorDict.keys())) )
	print("# keys in iVectorDict: " + str(len(iVectorDict)))
	iVectorsPDF = pd.DataFrame(list(iVectorDict.values()))

	#train test split
	iVectorsTrainPDF = iVectorsPDF[:-2] #train data
	iVectorsTestPDF = iVectorsPDF[-2:] #test data
	
	#fit principal components of dataset
	reducedPrincipalComponents, datasetMean = rxPCA.fit(iVectorsTrainPDF)

	#transform test data from image space to eigenspace
	rxPCA.transform(iVectorsTestPDF, reducedPrincipalComponents, datasetMean)

	plt.imshow(subImage, cmap='gray')

main()