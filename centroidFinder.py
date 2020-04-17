from joblib import load,dump
import numpy as np
pcDict = load("pcDict.bin")

# print(pcDict.keys())

ignoreKeys = ['useZeroMean', 'dim', 'facesCentroid']

try:
	del accumulator
except:
	pass
for key,value in pcDict.items():
	if key not in ignoreKeys:
		if 'accumulator' not in globals():
			accumulator = value
		else:
			accumulator = np.add(accumulator,value)
pcDict['facesCentroid'] = accumulator.tolist()
dump(pcDict, "pcDict.bin")