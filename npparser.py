import cPickle
import numpy as np
import csv

def parse(input_filename, labels_filename):
	session = numpy.load(open(input_filename, "rb"))[1:].astype(float)
	samples = []
	#Delete index row
	del session[0]

	increment = 0

	indices = np.where(session[:,-1] = 1)

	for i in range(len(indices)-1):
		newsample = session[range(indices[i], indices[i+1]), 1:-1].T
		newsample = newsample.tolist()
		newsample.append(increment)
		samples.append(newsample)
		increment += 1
	return samples
