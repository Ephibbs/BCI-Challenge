import csv
import cPickle
import numpy
import os
import sys


def convert(in_dir, out_dir):
    for file_name in os.listdir(in_dir):
        print "Processing: " + file_name

        numpy_arr = numpy.genfromtxt(in_dir + file_name, delimiter=",")

        numpy.save(out_dir + file_name.replace(".csv", ".npy"), numpy_arr)

    print "Finished converting."


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print "Please enter an argument of either \"train\" or \"test\"."
        sys.exit()

    if sys.argv[1] == "train":
        convert("./data/train/", "./data_processed/train/")
    elif sys.argv[1] == "test":
        convert("./data/test/", "./data_processed/test/")
