import cPickle
import numpy as np
import csv

def parse_visual(input_filename, output_filename):
	session = cPickle.load(open(input_filename, "rb"))

	#Delete index row
	del session[0]

	outf = open(output_filename + "-Pre.csv", "wb")
	writer = csv.writer(outf)
	increment = 1

	for i in session:
		if float(i[-1]) == 1:
			outf = open(output_filename + "seg" +  str(increment) + ".csv", "wb")
			writer = csv.writer(outf)
			writer.writerow(i)
			increment = increment + 1
		else:
			writer.writerow(i)



