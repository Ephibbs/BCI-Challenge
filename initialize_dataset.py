import csv
import numpy
import os


def convert(in_dir, out_dir, force=False):
    '''Converts datasets from .csv to .npy formats.

    Args:
        in_dir: Input directory of .csv formatted data.
        out_dir: Output directory of .npy formatted data.
    '''
    for file_name in os.listdir(in_dir):
        print "Processing: " + file_name

        if (file_name.replace(".csv", ".npy") in os.listdir(out_dir)) and not force:
            print "Already processed (skipping): " + file_name
            continue

        numpy_arr = numpy.genfromtxt(in_dir + file_name, delimiter=",")

        numpy.save(out_dir + file_name.replace(".csv", ".npy"), numpy_arr)

    print "Finished converting."


if __name__ == "__main__":
    '''Initializes dataset.
    '''
    # if len(sys.argv) <= 1:
    #     print "Please enter an argument of either \"train\" or \"test\"."
    #     sys.exit()

    convert("./data/train/", "./data/data_processed/train/")
    convert("./data/test/", "./data/data_processed/test/")
