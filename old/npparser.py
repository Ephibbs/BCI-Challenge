import cPickle
import numpy as np
from sklearn import preprocessing
import csv
import glob
from pywt import * 
from scipy.signal import filtfilt, butter
from getSamples import *

#Get Files
trainsamples, valsamples, testsamples, lengths = getSamples("./data/data_processed/train/*", "./data/data_processed/test/*", "./data/ydata.npy")
print len(trainsamples), len(trainsamples[0])
np.save("trainX.npy", trainsamples[0])
np.save("trainY.npy", trainsamples[1])
print "done with train"

np.save("valX.npy", valsamples[0])
np.save("valY.npy", valsamples[1])
print "done with val"

np.save("testX.npy", testsamples)
print "done with test"

np.save("trainlengths.npy", lengths[0])
np.save("vallengths.npy", lengths[1])
np.save("testlengths.npy", lengths[2])
print "done with lengths"