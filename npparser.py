import cPickle
import numpy as np
from sklearn import preprocessing
import csv
import glob

featuresize = 810
numdiscard = 0
trainfiles = glob.glob("../data/data_processed/train/*")
testfiles = glob.glob("../data/data_processed/test/*")
y = np.load("ydata.npy")
newy = []
#maxamplitudes = [[0]]*57
samples = []
discarded = []
for file in trainfiles:
	session = np.load(file)[1:].astype(float)
	#Delete index row
	np.delete(session, 0)

	increment = 0
	indices = np.where(session[:,-1] == 1)[0]

	for i in range(len(indices)):
		if indices[i]+featuresize < len(session) or (i+1 < len(indices) and indices[i]+featuresize < indices[i+1]):
			newsample = session[range(indices[i], indices[i]+featuresize), 1:-1].T
			#sample2 = np.array(newsample)
			#np.delete(newsample, -1)
			#np.delete(sample2, 0)
			#newsample -= sample2
			#newsample = newsample.T
			#newsample = newsample.tolist()
			#newsample.append(increment)
			newy.append(y[i])
			samples.append(newsample)
		else:
			numdiscard +=1
			#print session[range(indices[i], indices[i]+featuresize), 1:-1].shape
			# if i == len(indices)-1:
			# 	discarded.append(session[range(indices[i], len(session)), 1:-1].T)
			# else:
			# 	discarded.append(session[range(indices[i], indices[i+1]), 1:-1].T)
	# for sample in samples:
	# 	for j in range(len(sample)):
	# 		avg = np.sum(sample[j])/len(sample[j])
	# 		maxamp = np.amax(abs(sample[j]-avg))
	# 		if abs(maxamp) > maxamplitudes[j][0]:
	# 			maxamplitudes[j][0] = abs(maxamp)
	# 			print maxamp
print numdiscard
testsamples = []
for file in testfiles:
	session = np.load(file)[1:].astype(float)
	#Delete index row
	np.delete(session, 0)

	increment = 0
	indices = np.where(session[:,-1] == 1)[0]
	for i in range(len(indices)):
		if indices[i]+featuresize < len(session) or (i+1 < len(indices) and indices[i]+featuresize < indices[i+1]):
			newsample = session[range(indices[i], indices[i]+featuresize), 1:-1].T
			#sample2 = np.array(newsample)
			#np.delete(newsample, -1)
			#np.delete(sample2, 0)
			#newsample -= sample2
			#newsample = newsample.T
			#newsample = newsample.tolist()
			#newsample.append(increment)
			newy.append(y[i])
			testsamples.append(newsample)
		else:
			numdiscard +=1

samples = np.array(samples)
testsamples = np.array(testsamples)
trainshape = samples.shape
testshape = testsamples.shape
print trainshape, testshape
numsamples = trainshape[0] + testshape[0]
print numsamples
samples = preprocessing.scale(np.concatenate((samples, testsamples)).flatten())
samples = samples.reshape((numsamples, 57, featuresize))
train = samples[:trainshape[0]]
test = samples[trainshape[0]:]
print train.shape, test.shape
# # for file in files:
# # 	session = np.load(file)[1:].astype(float)
# # 	samples = []
# # 	#Delete index row
# # 	np.delete(session, 0)
# #
# # 	increment = 0
# #
# # 	indices = np.where(session[:,-1] == 1)[0]
# # 	for i in indices:
# # 		newsample = session[range(i, i+featuresize), 1:-1].T
# # 		newsample -= np.sum(newsample, axis=1).reshape(len(newsample),1)/featuresize
# # 		newsample = np.divide(newsample, maxamplitudes)
# # 		#newsample = newsample.tolist()
# # 		#newsample.append(increment)
# # 		samples.append(newsample)
# # 		increment += 1
# '''"normalized_data/"'''
# print numdiscard
np.save("testX.npy", test)
np.save("featuresX.npy", train)
np.save("featuresY.npy", newy)
np.save("discarded.npy", np.array(discarded))
