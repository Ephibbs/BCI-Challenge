import numpy as np
from sklearn import svm, metrics, ensemble, naive_bayes, linear_model, metrics, cross_validation
from sklearn.neural_network import BernoulliRBM
import sys
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from seqlearn.hmm import MultinomialHMM
from seqlearn.perceptron import StructuredPerceptron
from shuffle import *

def score(classifier, X, Y):
	fpr, tpr, thresholds = metrics.roc_curve(Y, classifier.predict(X), pos_label=1)
	auc = metrics.auc(fpr,tpr)
	return auc
	#pred = classifier.predict(X, lengths)
	#return 1 - np.sum(abs(pred - Y).astype(float))/len(pred)


def output_results(results, output_filename):
	f = open(output_filename, "w")
	for r in results:
	    f.write(str(r)+"\n")
	f.close()


def load_train_set(input_filename):
	x = np.load(input_filename + "X.npy")
	y = np.load(input_filename + "Y.npy")
	lengths = np.load(input_filename + "lengths.npy").astype(int)

	return x, y, lengths


def learn():
	trainx, trainy, trainlengths = load_train_set("train")
	classifier = ensemble.RandomForestClassifier()
	classifier.fit(trainx, trainy) #trainlengths
	#trainx, trainy = shuffle(trainx, trainy)
	print score(classifier, trainx, trainy)#, trainlengths)
	return classifier


def cross_validate(classifier):
	valX, valY, vallengths = load_train_set("val")
	#, vallengths)
	#print score(classifier, valX, valY, vallengths)
	#valX, valY = shuffle(valX, valY)
	pred = classifier.predict(valX)
	print score(classifier, valX, valY)
	print(metrics.classification_report(valY, pred))



def print_test(classifier):
	testx = np.load("testX.npy")
	testlengths = np.load("testlengths.npy").astype(int)

	results = classifier.predict(testx)
	output_results(results, "results.txt")

classifier = learn()

cross_validate(classifier)

print_test(classifier)
