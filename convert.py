import csv
import cPickle
import numpy
import os


# Convert all training-set .csv files into a numpy binary data file
for file_name in os.listdir("./data/train"):
	print "Processing: " + file_name

	numpy_arr = numpy.genfromtxt("./data/train/" + file_name, delimiter=",")

	numpy.save("./data_processed/" + file_name.replace(".csv", ".npy"), numpy_arr)
