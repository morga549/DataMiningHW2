# A skeleton for implementing Naive Bayes Classifier in Python.
# Author: Salem

import numpy
import scipy.stats as stats
import sys

trainingFile = sys.argv[1]
testingFile = sys.argv[2]
Xtrain = numpy.loadtxt(trainingFile)

# n = number of training points, d = dimensions of training points
n = Xtrain.shape[0]
d = Xtrain.shape[1] - 1

print(str(n))
print(str(d))

# Sort training data by class
sortedArr = Xtrain[Xtrain[:, d].argsort()]

# Make separate arrays for each class
index = 0
i = 0
while i < len(sortedArr):
    if sortedArr[i][d] == -1 and sortedArr[i + 1][d] == 1:
        index = i
    i += 1



# split sorted array into classes
neg = sortedArr[:index]
pos = sortedArr[index:]

#print(neg)
#print("\n\n ..................\n\n")
#print(pos)

# flip axis to make calculating mean and standard deviation easierf
neg = numpy.swapaxes(neg, 0, 1)
pos = numpy.swapaxes(pos, 0, 1)

# Training... Collect mean and standard deviation for each dimension for each class..
# Also, calculate P(C+) and P(C-)

posMeans = []
posSTDs = []

negMeans = []
negSTDs = []

# calculate means and standard deviations for each attribute in positive and negative matrix
for x in range(0, d):
    posMeans.append(numpy.mean(pos[x]))
    posSTDs.append(numpy.std(pos[x]))

    negMeans.append(numpy.mean(neg[x]))
    negSTDs.append(numpy.std(neg[x]))

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

# iterate points in test data
for x in range(0, nn):
    # scores for probability of point falling in positive class and negative class
    posMult = 1.0
    negMult = 1.0
    # class: 1 = positive; -1 = negative
    result = 0
    # iterate attributes in test point
    for y in range(0, d):
        # joint probability of attributes
        posMult *= stats.norm.pdf(Xtest[x][y], posMeans[y], posSTDs[y])
        negMult *= stats.norm.pdf(Xtest[x][y], negMeans[y], negSTDs[y])
    # weight joint probability of attributes by prior probability of each class
    posMult *= posPrior
    negMult *= negPrior

    # our prediction for class of test point
    if posMult > negMult:
        result = 1
    else:
        result = -1

    # actual class of test point
    actual = Xtest[x][4]

    # comparing our prediction to actual
    if result == 1 and actual == 1:
        tp += 1
    elif result == 1 and actual == -1:
        fp += 1
    elif result == -1 and actual == -1:
        tn += 1
    elif result == -1 and actual == 1:
        fn += 1

print("\nClassification Accuracy: " + str('{accuracy:.2%}'.format(accuracy=(float((tp + tn ))/( tp + tn + fp + fn)))))


print("True Positive: " + str(tp))
print("True Negative: " + str(tn))
print("False Positive: " + str(fp))
print("False Negative: " + str(fn))

print("Classification Precision: " + str(float(tp) / (tp + fp)))
print("Classification Recall: " + str(float(tp) / (tp + fn)))


