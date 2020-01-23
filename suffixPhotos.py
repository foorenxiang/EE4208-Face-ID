import os
import glob

suffix = '_Matric'
ext = ".jpg"

targetPath = "./"
destPath = "./renamed/"
files = glob.glob(targetPath+"*."+ext)
print(files)

for file in files:
	currentFile = open(file, 'rb').read()
	newFileName = file[:-len(ext)] + suffix + ext
	newFile = open(destPath+newFileName, 'wb+')
	os.remove(file)