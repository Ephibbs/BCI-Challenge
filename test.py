import numpy as np
from sklearn import svm
import random
import sys

trainx = np.load("featuresX.npy")
trainy = np.load("featuresY.npy")
testx = np.load("testX.npy")

newx = []
newtest = []

for i in trainx:
    newx.append(i[:4].flatten())

for i in testx:
    newtest.append(i[:4].flatten())

data = zip(newx, trainy)
random.shuffle(data)
trainx, trainy = zip(*data)
trainx = np.array(trainx)
trainy = np.array(trainy)
print trainy[:30]
#newx = np.array(newx)
#SVMs = []
SVM = svm.SVC()
# for i in range(len(x[0])):
#     s = svm.SVC()
#
#     s.fit(x[:len(x)/2,i], y[:len(x)/2])
#     #SVMs.append(s)
#     print i, s.score(x[len(x)/2:,i], y[len(x)/2:])

print "train"
#
# for i in range(len(x[0])):
#     SVMs[i].fit(x[:len(x)/2,i], y[:len(x)/2])

SVM.fit(trainx[:len(trainx)/2], trainy[:len(trainx)/2])
#print trainy[:30]
print "test"
print SVM.score(trainx[len(trainx)/2:], trainy[len(trainx)/2:])

SVM.fit(trainx, trainy)
f = open("results.txt", "w")
results = SVM.predict(newtest)
for r in results:
    f.write(str(r)+"\n")
# for i in range(len(x[0])):
#     print i, SVMs[i].score(x[len(x)/2:,i], y[len(x)/2:])
