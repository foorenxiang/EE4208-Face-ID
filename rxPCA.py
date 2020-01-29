import numpy as np
import pandas as pd
from numpy.linalg import eig
import pprint
pp = pprint.PrettyPrinter(indent=4)

class PCA:
	def __init__(self, n_components = 1):
		self.coVarianceExplanation = n_components
		self.reducedPrincipalComponents = None
		self.datasetMean = None
		self.isFitted = False

	#dataset must be numpy array
	def fit(self, dataset):
		if not isinstance(dataset, pd.DataFrame):
			raise Exception('inputPDF should be a pandas DataFrame!')
		#find mean of each feature
		#deduct mean of each feature from dataset
		self.datasetMean = np.mean(dataset, axis=0)
		dataset = dataset - self.datasetMean

		#find covariance matrix
		covMatrix = dataset.cov()

		#calculate eigen decomposition of covariance matrix (sort eigenvalues in descending order)
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
		
		varianceAggregator = 0
		componentsToKeep=0
		while (varianceAggregator/eigenValuesSum) < self.coVarianceExplanation:
			varianceAggregator+=principalComponents[componentsToKeep]['absEigValue']
			componentsToKeep+=1

		self.reducedPrincipalComponents = principalComponents[:componentsToKeep]

		# print('Components to keep: ' + str(componentsToKeep))
		# print("reducedPrincipalComponent: ")
		# pp.pprint(reducedPrincipalComponents)

		self.isFitted = True

	def transform(self, inputPDF):
		if not self.isFitted:
			raise Exception('PCA model is not fitted yet! Run PCA.fit() first.')
		if not isinstance(inputPDF, pd.DataFrame):
			raise Exception('inputPDF should be a pandas DataFrame!')

		#normalise test data
		inputPDF -= self.datasetMean
		#project test data to eigenspace
		projValues = []
		# print('inputPDF' + str(inputPDF))

		projValues = []
		for component in self.reducedPrincipalComponents:
			#use 0 index as dot product returns array
			projValue = component['eigVector'].dot(inputPDF)[0] 
			projValues.append(projValue)

		# print("\n\nprojValues: ")
		# pp.pprint(projValues)
		return projValues