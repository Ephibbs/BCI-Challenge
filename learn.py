from sklearn import svm
import csv
import cPickle
import random

SVM = svm.SVC()
deltas = cPickle.load(open("deltas.pkl", "rb"))
testDeltas = cPickle.load(open("../test/deltas.pkl", "rb"))

Y = csv.reader(open("../TrainLabels.csv"), delimiter=",")

newX = []
newY = []
testX = []

for delta in deltas:
	newX.append([float(delta)])

for delta in testDeltas:
	testX.append([float(delta)])

for iteration, i in enumerate(Y):
	if iteration == 0:
		continue
	else:
		newY.append(int(i[1]))

#scramble
data = zip(newX, newY)
random.shuffle(data)
newX, newY = zip(*data)

half = len(newX) / 2

SVM.fit(newX[:half], newY[:half])
print SVM.score(newX[half:], newY[half:])

prediction = SVM.predict(testX)

f = open("predictions.csv", "w")
for p in prediction:
	f.write(str(p)+"\n")