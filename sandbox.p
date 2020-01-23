def calcFibSeq(x):
	fibSeq = [1,1]
	while len(fibSeq)<x:
		fibSeq.append(fibSeq[-1] + fibSeq[-2])
	return fibSeq[-1],fibSeq[:x]

lastNum, fibSeq = calcFibSeq(100)
print(lastNum)
print(fibSeq)