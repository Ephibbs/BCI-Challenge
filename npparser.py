import cPickle
import numpy as np
import csv
import glob

featuresize = 500

files = glob.glob("../data/data_processed/*")
maxamplitudes = [[0]]*57
for file in files:
	session = np.load(file)[1:].astype(float)
	samples = []
	#Delete index row
	np.delete(session, 0)

	increment = 0
	indices = np.where(session[:,-1] == 1)[0]

	for i in indices:
		newsample = session[range(i, i+featuresize), 1:-1]
		sample2 = np.array(newsample)
		np.delete(newsample, -1)
		np.delete(sample2, 0)
		newsample -= sample2
		newsample = newsample.T
		#newsample = newsample.tolist()
		#newsample.append(increment)
		samples.append(newsample)
		increment += 1
	# for sample in samples:
	# 	for j in range(len(sample)):
	# 		avg = np.sum(sample[j])/len(sample[j])
	# 		maxamp = np.amax(abs(sample[j]-avg))
	# 		if abs(maxamp) > maxamplitudes[j][0]:
	# 			maxamplitudes[j][0] = abs(maxamp)
	# 			print maxamp

# for file in files:
# 	session = np.load(file)[1:].astype(float)
# 	samples = []
# 	#Delete index row
# 	np.delete(session, 0)
#
# 	increment = 0
#
# 	indices = np.where(session[:,-1] == 1)[0]
# 	for i in indices:
# 		newsample = session[range(i, i+featuresize), 1:-1].T
# 		newsample -= np.sum(newsample, axis=1).reshape(len(newsample),1)/featuresize
# 		newsample = np.divide(newsample, maxamplitudes)
# 		#newsample = newsample.tolist()
# 		#newsample.append(increment)
# 		samples.append(newsample)
# 		increment += 1
	'''"normalized_data/"'''
	np.save("delta_data/"+file.split("/")[-1], samples)
