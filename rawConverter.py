from PIL import Image
import glob

filePath = "./"
fileExt = ".raw"
files = glob.glob(filePath+"*"+fileExt)
files = list(map(lambda file:file[len(filePath):], files))

def rawToMatrix(rawData, width, height):
	rawData=[4:] #erase resolution data from image buffer
	imageMatrix=[]
	for row in range(height):
		startIndex = row*width
		endIndex = startIndex+width
		imageMatrix.append(rawData[startIndex:endIndex])

for filename in files:
	print(filename)
	file = open(filename, 'rb')
	rawData = file.read()
	
	width = rawData[0] + (rawData[1] << 8)
	height = rawData[2] + (rawData[3] << 8)
	print(width, height)
	
	imgSize = (width, height)# the image size
	img = Image.frombytes('L', imgSize, rawData)
	img.save(filePath+filename[:-3]+"jpg")# can give any format you like .png