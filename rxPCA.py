import numpy as np
import pandas as pd
from numpy.linalg import eig
import pprint
pp = pprint.PrettyPrinter(indent=4)

#dataset must be numpy array
def fit(dataset):
	#find mean of each feature
	#deduct mean of each feature from dataset
	datasetMean = np.mean(dataset, axis=0)
	dataset = dataset - datasetMean

	#find covariance matrix
	covMatrix = dataset.cov()

	#calculate eigen decomposition of covariance matrix (sort eigenvalues in descending order)
	# covMatrixDet = determinants(covMatrix)
	eigValues, eigVectors = eig(covMatrix)
	absEigValues = list(map(lambda x: abs(x), eigValues))
	principalComponents = []
	for eigValue, absEigValue, eigVector in sorted(zip(eigValues, absEigValues, eigVectors), key=lambda x: x[1], reverse = True):
		principalComponents.append({
			'eigValue': eigValue,
			'absEigValue': absEigValue,
			'eigVector': eigVector
			})

	eigenValuesSum = sum(absEigValues)

	varianceExplanation = 0.7
	
	varianceAggregator = 0
	componentsToKeep=0
	while (varianceAggregator/eigenValuesSum) < varianceExplanation:
		varianceAggregator+=principalComponents[componentsToKeep]['absEigValue']
		componentsToKeep+=1

	# print('Components to keep: ' + str(componentsToKeep))
	reducedPrincipalComponents = principalComponents[:componentsToKeep]

	# print("reducedPrincipalComponent: ")
	# pp.pprint(reducedPrincipalComponents)

	return reducedPrincipalComponents, datasetMean

def transform(iVectorsTestPDF, reducedPrincipalComponents, datasetMean):
	#normalise test data
	iVectorsTestPDF -= datasetMean
	#project test data to eigenspace
		#iterate through each dataframe row
	projValues = []
	for index, row in iVectorsTestPDF.iterrows():
		projValue = []
		for component in reducedPrincipalComponents:
			projValueDim = component['eigVector'].dot(row)
			projValue.append(projValueDim)
		projValues.append(projValue)
	print("\n\nprojValues: ")
	pp.pprint(projValues)
	return projValues
