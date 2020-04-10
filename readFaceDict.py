import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils

import numpy as np
import pandas as pd
from joblib import dump, load

faceDict = load('faceDict.bin')
try:
	plt.figure(figsize=(8,8))
	plt.imshow(faceDict['foo ren xiang'], cmap='gray')
	plt.show()
except KeyError:
	pass
