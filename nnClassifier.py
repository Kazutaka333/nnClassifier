import numpy as np
import pprint
import math
# Load data set and code labels as 0 = 'NO', 1 = 'DH', 2 = 'SL'
labels = ["NO", "DH", "SL"]
data = np.loadtxt("column_3C.dat", converters={6: lambda s: labels.index(s)})
# Separate features from labels
x = data[:,0:6]
y = data[:,6]
# Divide into training and test set
training_indices = range(0,80) + range(100,148) + range(160,280)
test_indices = range(80,100) + range(148,160) + range(280,310)
trainx = x[training_indices,:]
trainy = y[training_indices]
testx = x[test_indices,:]
testy = y[test_indices]
l1Prediction = []
l2Prediction = []

for i in range(len(testx)):
    factors = testx[i]
    shortestL1 = None
    l1Label = None
    shortestL2 = None
    l2Label = None
    # calculate l1 and l2
    for j in range(len(trainx)):
        trainFactors = trainx[j]
        trainLabel = trainy[j]
        l1 = 0
        l2 = 0
        for k in range(6):
            l1 += abs(factors[k] - trainFactors[k])
            l2 += (factors[k] - trainFactors[k])**2
        l2 = math.sqrt(l2)

        if (shortestL1 is None) or shortestL1 > l1:
            shortestL1 = l1
            l1Label = trainLabel
        if shortestL2 is None or shortestL2 > l2:
            shortestL2 = l2
            l2Label = trainLabel

    l1Prediction.append(l1Label)
    l2Prediction.append(l2Label)

# count wrong label
l1ErrorCount = 0
l2ErrorCount = 0
matrixSize = (3, 3)
l1Matrix = np.zeros(matrixSize, dtype=np.int)
l2Matrix = np.zeros(matrixSize, dtype=np.int)
for i in range(len(testy)):
    correctLabel = testy[i]
    l1Matrix[int(correctLabel)][int(l1Prediction[i])] += 1
    l2Matrix[int(correctLabel)][int(l2Prediction[i])] += 1
    if l1Prediction[i] != correctLabel:
        l1ErrorCount += 1
    if l2Prediction[i] != correctLabel:
        l2ErrorCount += 1
testSize = len(testy)
print l1ErrorCount
l1ErrorRate = float(l1ErrorCount) / float(testSize)
l2ErrorRate = float(l2ErrorCount) / float(testSize)
print "l1ErrorRate"
print l1ErrorRate
pprint.pprint(l1Matrix)
print "l2ErrorRate"
print l2ErrorRate
pprint.pprint(l2Matrix)
