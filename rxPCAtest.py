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

from rxPCA import PCA

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
	pcaModel = PCA(n_components=0.7)
	pcaModel.fit(iVectorsTrainPDF)

	#transform test data from image space to eigenspace
	pcaModel.transform(iVectorsTestPDF)

	plt.imshow(subImage, cmap='gray')

main()