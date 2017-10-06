import numpy
import scipy.stats as stats
import operator
import functools
import sys

trainingFile = sys.argv[1]
testingFile = sys.argv[2]
Xtrain = numpy.loadtxt(trainingFile)

def compareProbs(a):
    if a[0] > a[1]: return 1
    else: return -1

# n = number of training points, d = dimensions of training points
n = Xtrain.shape[0]
d = Xtrain.shape[1] - 1

posFlag = Xtrain[:, d] > 0      # indices of C+ records
pos = Xtrain[posFlag, 0: d]     # subarray w/ just C+ records
negFlag = Xtrain[:, d] < 0      # indices of C- records
neg = Xtrain[negFlag, 0: d]     # subarray w/ just C- records

# calculate means and standard deviations for each attribute in positive and negative matrix
posMean = numpy.mean(pos, axis=0)
negMean = numpy.mean(neg, axis=0)

posSTD = numpy.std(pos, axis=0)
negSTD = numpy.std(neg, axis=0)

# calculate prior probability for classes
posPrior = float(len(pos)) / len(Xtrain)
negPrior = float(len(neg)) / len(Xtrain)

# Testing .....
Xtest = numpy.loadtxt(testingFile)
nn = Xtest.shape[0]  # Number of points in the testing data.

tp = 0  # True Positive
fp = 0  # False Positive
tn = 0  # True Negative
fn = 0  # False Negative

posJoint = stats.norm.pdf(Xtest[:, 0:d], posMean, posSTD)
negJoint = stats.norm.pdf(Xtest[:, 0:d], negMean, negSTD)

posPredict = [functools.reduce(operator.mul, x, posPrior) for x in posJoint]
negPredict = [functools.reduce(operator.mul, x, negPrior) for x in negJoint]

bothPredict = list(zip(posPredict, negPredict))

results = list(zip(map(compareProbs, bothPredict), Xtest[:,d]))

for x,y in results:
    if(x == y == 1): tp += 1
    elif(x == y == -1): tn += 1
    elif(x == 1): fp += 1
    elif(x == 0): fn += 1

print('Accuracy: ' + str(float(tp + tn) / (len(results))))

print('tp: ' + str(tp))
print('tn: ' + str(tn))
print('fp: ' + str(fp))
print('fn: ' + str(fn))

print('Precision: ' + str(float(tp)/(tp+fp)))
print('Recall: ' + str(float(tp)/(tp+fn)))
