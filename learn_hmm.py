import csv
import numpy as np
from seqlearn.hmm import MultinomialHMM

# Load in data
dat    = np.load("./data_processed/Data_S02_Sess01.npy")
labels = csv.reader(open("./data/TrainLabels.csv"), delimiter=",")

# Create the X
X = dat
X = np.delete(X, (0), axis=0) # delete first row (labels)
X = np.delete(X, (0), axis=1) # delete first column (Time -- label)
X_width = X.shape[1] # width of the array
X = np.delete(X, (X_width - 1), axis=1) # delete last column (FeedBackEvent -- label)

# Create the y


# decode : Decoding algorithm
# alpha  : Lidstone (additive) smoothing parameter
clf = MultinomialHMM(decode='viterbi', alpha=0.01) 

# fit(X, y, lengths)
#     Parameters:
#     -----------
#     X : {array-like, sparse matrix}, shape (n_samples, n_features)
#         Feature matrix of individual samples.
#     y : array-like, shape (n_samples,)
#         Target labels.
#     lengths : array-like of integers, shape (n_sequences,)
#         Lengths of the individual sequences in X, y. The sum of these should be n_samples.
#
#     Returns:    
#     --------
#     self : MultinomialHMM
# clf.fit()
