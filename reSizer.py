import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
import numpy as np
import pandas as pd
from joblib import dump, load
import glob

font = cv2.FONT_HERSHEY_SIMPLEX
path = "./"
fileExt = ".tif"
files = glob.glob(path+"*"+fileExt)

#resize subimage
dim = (100, 100) #width, height

for file in files:
	file = file[2:]
	gray = cv2.imread(file,0)
	gray = cv2.resize(gray, dim)
	fileLoc = file[:-4] + '.jpg'
	print(gray)
	print(fileLoc)
	cv2.imwrite(fileLoc, gray)