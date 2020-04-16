from joblib import load, dump
import sys
if len(sys.argv) == 1:
	print("Enter dictionary file as script argument!")
	sys.exit()
pcDict = load(sys.argv[1])
print("Dictionary pre-adjustment size:" + str(len(pcDict)))
keys = list(pcDict.keys())
print(keys)
print("Removed keys:")
for key in keys:
	for i in range(18, 79):
		if "person"+str(i) in key:
			print(key)
			del pcDict[key]
print("Dictionary post-adjustment size:" + str(len(pcDict)))
dump(pcDict, sys.argv[1].split('.')[0] + "_Adjusted." + sys.argv[1].split('.')[1])