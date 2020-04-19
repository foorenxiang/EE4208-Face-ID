import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import time

def strFloat(floatVal):
	return "{0:.2f}".format(round(floatVal,2))

def EVD(X):
    # s, U = eigsh(X,k=X.shape[0])
    s, U = eigh(X, turbo = True, overwrite_a = True, overwrite_b = True)
    idx = s.argsort()[::-1] # decreasing order
    return s[idx], U[:,idx]

class PCA:
	def __init__(self, n_components = 0.9):
		self.coVarianceExplanation = n_components
		self.reducedPrincipalComponents = None
		self.datasetMean = None
		self.isFitted = False

	#dataset must be numpy array
	#each subImage is stored as a row in the pandas dataframe
	#each pixel of a subImage is stored in a series
	def fit(self, dataset):
		if not isinstance(dataset, pd.DataFrame):
			raise Exception('inputPDF should be a pandas DataFrame!')
		#find mean of each feature
		#deduct mean of each feature from dataset
		# self.datasetMean = np.mean(dataset, axis=0)
		# dataset = dataset - self.datasetMean

		dataset=pd.DataFrame(dataset).to_numpy()
		dataset -= dataset.mean(axis=0)
		dataset /= np.std(dataset,axis=0)
		covMatrix = dataset.T.dot(dataset) / dataset.shape[0]
		t = time.time()
		eigValues, eigVectors = EVD(covMatrix)
		elapsed = time.time() - t
		print("Eigendecomposition time: " + str(elapsed) +' seconds')
		eigValues, eigVectors = np.real(eigValues), np.real(eigVectors)

		print("eigVectors.shape")
		print(eigVectors.shape)
		absEigValues = eigValues
		principalComponents = []
		for eigValue, absEigValue, eigVector in sorted(zip(eigValues, absEigValues, eigVectors), key=lambda x: x[1], reverse = True):
			principalComponents.append({
				'eigValue': eigValue,
				'absEigValue': absEigValue,
				'eigVector': eigVector
				})

		eigenValuesSum = sum(absEigValues)

		print("absEigValues")
		
		componentsToKeep=0
		if self.coVarianceExplanation<float(1):
			varianceAggregator = 0
			while (varianceAggregator/eigenValuesSum) < self.coVarianceExplanation:
				varianceAggregator+=principalComponents[componentsToKeep]['absEigValue']
				print(principalComponents[componentsToKeep]['absEigValue'])
				componentsToKeep+=1
			print("With " + str(self.coVarianceExplanation*100) + "% covariance explanation specified, " + str(componentsToKeep) + " principal components are required")
		else:
			componentsToKeep = int(self.coVarianceExplanation)

			totalEigValuesUsed = sum(principalComponent['absEigValue'] for principalComponent in principalComponents[:self.coVarianceExplanation])

			totalEigValuesPresent = sum(principalComponent['absEigValue'] for principalComponent in principalComponents)

			print("With " + str(componentsToKeep) + " principal components specified, " + strFloat(totalEigValuesUsed/totalEigValuesPresent*100) + "% covariance is explained")

		self.reducedPrincipalComponents = principalComponents[:componentsToKeep]

		self.isFitted = True

	def transform(self, inputPDF):
		if not self.isFitted:
			raise Exception('PCA model is not fitted yet! Run PCA.fit() first.')
		if not isinstance(inputPDF, pd.DataFrame):
			print('inputPDF type: ' + str(type(inputPDF)))
			raise Exception('inputPDF should be a pandas DataFrame!')

		#normalise test data
		inputPDF=pd.DataFrame(inputPDF).to_numpy()
		inputPDF -= inputPDF.mean(axis=0)
		#check if inputPDF should be transposed
		try:
			self.reducedPrincipalComponents[0]['eigVector'].dot(inputPDF)[0]
		except ValueError:
			inputPDF = inputPDF.T

		if np.isnan(inputPDF).any():
			print("inputPDF NAN")
			raise ValueError
		projValues = []
		for component in self.reducedPrincipalComponents:
			#use 0 index as dot product returns array
			projValue = component['eigVector'].dot(inputPDF)[0]
			if np.isnan(projValue).any():
				print("projValue NAN")
				raise ValueError
			projValues.append(projValue)

		return projValues