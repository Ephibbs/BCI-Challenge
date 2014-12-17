import cPickle
import numpy as np
from sklearn import preprocessing
import csv
import glob
from pywt import * 
from scipy.signal import filtfilt, butter
from getSamples import *

def splitUpSamples(files):
	X = []
	lengths = []
	for f in files:
		session = np.load(f)[1:].astype(float)
		indices = np.where(session[:,-1] == 1)[0]
		cntr = 0
		for i in range(len(indices)):
			stop = indices[i+1] if i+1 < len(indices) else len(session)
			channels = session[range(indices[i], stop), 1:-1].T
			sample = butterAndWavelet(channels)
			X.append(sample)
			cntr += 1
		lengths.append(cntr)
	return (X, lengths)

def getSamples(traindirpath, testdirpath, labelspath):
	alltrainfiles = glob.glob(traindirpath)
	alltrainLabels = np.load(labelspath)

	trainfiles = alltrainfiles[:len(alltrainfiles)/2]
	trainSamples, trainLengths = splitUpSamples(trainfiles)
	trainLabels = alltrainLabels[:len(trainSamples)]
	print "training files"

	valfiles = alltrainfiles[len(alltrainfiles)/2:]
	valSamples, valLengths = splitUpSamples(valfiles)
	valLabels = alltrainLabels[len(trainSamples):]
	print "val files"

	testfiles = glob.glob(testdirpath)
	testSamples, testLengths = splitUpSamples(testfiles)
	print "testing files"

	return ((trainSamples, trainLabels), (valSamples, valLabels), testSamples, (trainLengths, valLengths, testLengths))

def butterAndWavelet(channels):
	N = 2
	Wn = 0.01
	B, A = butter(N, Wn)
	sample = np.array([dwt(filtfilt(B, A, channel), 'haar') for channel in channels]).flatten()
	return sample