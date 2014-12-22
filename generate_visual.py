import numpy as np
import csv
from scipy.signal import filtfilt, butter
from sklearn import preprocessing

def parse_to_visual(input_filename):
	#Get rid of time column
	session = np.load(input_filename)[1:].astype(float)

	#Get indices
	indices = np.where(session[:,-1] == 1)[0]

	output_table = []
	firstRound = True
	for i in indices:
		#Get specific channel for specific signal
		channelFCz = session[range(i, i + 1001), 20]
		#Butters
		sample = butterSignal(channelFCz)
		sample = preprocessing.scale(sample)
		#concatenate
		if firstRound:
			output_table = sample
			firstRound = False
		else:
			output_table = np.vstack((output_table, sample))
		print output_table.shape

	return output_table.T

def butterSignal(channel):
	N = 2
	Wn = .05
	B, A = butter(N, Wn)
	return filtfilt(B, A, channel)

visual_arr = parse_to_visual("./data/data_processed/train/Data_S02_Sess01.npy")
print visual_arr.shape

out_writer = csv.writer(open("visual_arr_scaled.csv", 'wb'))

for row in visual_arr:
	out_writer.writerow(row)
