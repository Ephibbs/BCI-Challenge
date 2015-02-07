## Adapted from phalaris's GBM benchmark. Thanks phalaris.

import time
import sys
import numpy as np
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn import metrics
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from scipy.signal import filtfilt, butter
from time import time
from sklearn.decomposition import RandomizedPCA
import pandas as pd
from sklearn.cross_validation import cross_val_score
from parseData import *
from score import *

def predictions_to_file(algo, train, train_labels, redo, channels, start, end):
    test = parse_data(redo, "test", channels, start, end, 3400)
    algo.fit(train, train_labels)
    preds = algo.predict_proba(test) # why
    preds = preds[:,1]
    submission['Prediction'] = preds
    submission.to_csv('prediction.csv',index=False)


if __name__ == "__main__":

    allSensors = ['EOG', 'FCz', 'O2', 'O1', 'Fz', 'TP7', 'CPz', 'C3', 'C2', 'C1', 'C6', 'C5', 'TP8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Cz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'POz', 'Pz', 'C4', 'P4', 'FT7', 'FT8', 'AF8', 'PO7', 'AF4', 'AF3', 'P2', 'P3', 'P1', 'P6', 'P7', 'T8', 'P5', 'T7', 'P8', 'Fp1', 'Fp2', ' AF7', 'P08']

    start_time = time()

    submission = pd.read_csv('data/SampleSubmission.csv')
    #Good: ['C2', 'CPz', 'FC4', 'C1', 'FC2', 'Cz', 'CP4', 'C4', 'Pz', 'FT8', 'P2', 'CP6', 'CP2', 'P4']

    sensors = allSensors
    start = 30
    end = 150

    print 'Getting data'

    train = parse_data("train", sensors, start, end, 5440)

    #+==========PCA============+
    # n_components = 20

    # print("Extracting the top %d eigenfaces from %d faces"
    #       % (n_components, train.shape[0]))
    # t0 = time()
    # pca = RandomizedPCA(n_components=n_components, whiten=True).fit(train)
    # print("done in %0.3fs" % (time() - t0))

    # #eigenfaces = pca.components_.reshape((n_components, h, w))

    # print("Projecting the input data on the eigenfaces orthonormal basis")
    # t0 = time()
    # train = pca.transform(train)
    # print("done in %0.3fs" % (time() - t0))

    train_labels = pd.read_csv('data/TrainLabels.csv').values[:,1].ravel().astype(int)

    print 'Training GBM'

    ###     Algorithms      ###
    #algo = AdaBoostClassifier(n_estimators=750, learning_rate=0.05)
    algo = GradientBoostingClassifier(n_estimators=750, learning_rate=0.05, subsample=0.9)
    #algo = SGDClassifier()
    #algo = SVC(gamma=2, C=1)
    #algo = QDA()
    #algo = LDA()
    #algo = GaussianNB()
    #algo = RandomForestClassifier(max_depth=5, n_estimators=1000, max_features=100)

    # Print prediction report and scores
    score(algo, train, train_labels)

    # Make predictions and save to file
    #predictions_to_file(algo, train, train_labels, redo, sensorstouse, start, end)

    print "Finished."
    print "Running time: " + str(time() - start_time)


