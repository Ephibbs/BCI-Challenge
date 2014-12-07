import csv
import cPickle
# import numpy
import os


# Convert all training-set .csv files into pickles
for file_name in os.listdir("./data/train"):
	print "Processing: " + file_name

	# Code to pickle into a numpy array

	# numpy_arr = numpy.genfromtxt("./data/train/" + file_name, delimiter=",")
	# pickle = open("./data_processed/" + file_name.replace(".csv", ".pkl"), "w+b")

	# cPickle.dump(numpy_arr, pickle)

	data_list = list(csv.reader(open("./data/train/" + file_name), delimiter=","))
	pickle_file = open("./data_processed/" + file_name.replace(".csv", ".pkl"), "w+b")

	cPickle.dump(data_list, pickle_file)


# import csv
# import cPickle

# finaloutput = []

# for filen in range(1, 27):
# 	for sessn in range(1, 6):
# 		try:
# 			session = csv.reader(open("./data/Data_S" + "{0:02d}".format(filen) + "_Sess0" + str(sessn) + ".csv"), delimiter=",")
# 		except IOError:
# 			continue

# 		output = []
# 		low = -10000000
# 		high = 10000000
# 		for iteration, i in enumerate(session):
# 			if iteration == 0:
# 				continue
# 			if float(i[-1]) == 1:
# 				output.append(high - low)
# 				high = -10000000
# 				low = 10000000
# 			else:
# 				if float(i[1]) < low:
# 					low = float(i[1])
# 				if float(i[1]) > high:
# 					high = float(i[1])

# 		output.append(high - low)
# 		del output[0]
# 		finaloutput.extend(output)

# print len(finaloutput)

# outf = open("deltas.pkl", "wb")
# cPickle.dump(finaloutput, outf)
# #writer = csv.writer(outf)
# #writer.writerow(finaloutput)
