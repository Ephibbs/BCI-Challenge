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

def score(classifier, X, y):
    fpr, tpr, thresholds = metrics.roc_curve(y, classifier.predict(X), pos_label=1)
    auc = metrics.auc(fpr,tpr)
    return auc


def get_column(np_array, label):
    '''Returns the column corresponding to the column label in the array. Specific
    to our EEG data.

    Args:
        np_array: the array
        label: the column label
    '''
    column = {'Time':0, 'Fp1':1, 'Fp2':2, ' AF7':3, 'AF3':4, 'AF4':5, 'AF8':6, 'F7':7, 'F5':8, 'F3':9, 'F1':10, 'Fz':11, 'F2':12, 'F4':13, 'F6':14, 'F8':15, 'FT7':16, 'FC5':17, 'FC3':18, 'FC1':19, 'FCz':20, 'FC2':21, 'FC4':22, 'FC6':23, 'FT8':24, 'T7':25, 'C5':26, 'C3':27, 'C1':28, 'Cz':29, 'C2':30, 'C4':31, 'C6':32, 'T8':33, 'TP7':34, 'CP5':35, 'CP3':36, 'CP1':37, 'CPz':38, 'CP2':39, 'CP4':40, 'CP6':41, 'TP8':42, 'P7':43, 'P5':44, 'P3':45, 'P1':46, 'Pz':47, 'P2':48, 'P4':49, 'P6':50, 'P8':51, 'PO7':52, 'POz':53, 'P08':54, 'O1':55, 'O2':56, 'EOG':57, 'FeedBackEvent':58}

    return np_array[:, column[label]]


def fft(time_data):
    '''Computes magnitude of a fast fourier transform, by frequency, on a log10 scale.

    Args:
        time_data: Time-series data.

    Returns:
        A list of magnitudes corresponding to the signal frequencies falling in certain ranges.
    '''

    transform = np.fft.rfft(time_data)

    sliced = transform[1:48] # 1Hz, 2Hz, ... , 47Hz

    return np.log10(np.absolute(sliced))



def parse_data(dirName, channel, start, end, samples):
    '''Generates X and y to be fed into our predictor, and saves them to disk.

    Args:
        dirName: The directory name holding the data (e.g., "train", "test")
        subs: The subjects.
        channels: The EEG data channels to use.
        duration: The number of time-intervals, per sample, to use.
    '''

    # column = {'Time':0, 'Fp1':1, 'Fp2':2, ' AF7':3, 'AF3':4, 'AF4':5, 'AF8':6, 'F7':7, 'F5':8, 'F3':9, 'F1':10, 'Fz':11, 'F2':12, 'F4':13, 'F6':14, 'F8':15, 'FT7':16, 'FC5':17, 'FC3':18, 'FC1':19, 'FCz':20, 'FC2':21, 'FC4':22, 'FC6':23, 'FT8':24, 'T7':25, 'C5':26, 'C3':27, 'C1':28, 'Cz':29, 'C2':30, 'C4':31, 'C6':32, 'T8':33, 'TP7':34, 'CP5':35, 'CP3':36, 'CP1':37, 'CPz':38, 'CP2':39, 'CP4':40, 'CP6':41, 'TP8':42, 'P7':43, 'P5':44, 'P3':45, 'P1':46, 'Pz':47, 'P2':48, 'P4':49, 'P6':50, 'P8':51, 'PO7':52, 'POz':53, 'P08':54, 'O1':55, 'O2':56, 'EOG':57, 'FeedBackEvent':58}

    # for key, val in column.iteritems():
    # if val == channel:
    #     print name

    print '========loading '+dirName+' data========'

    data = np.load(open("data/"+dirName+"_no_pad.npy", "rb"))
    print data.shape

    data = data[:,:,31,30:150]
    print data
    data = data.reshape((16*340,120))
    print "\n\n\n"
    print data

    return data

def butterAndWavelet(channel):
    N = 2
    Wn = 0.01
    B, A = butter(N, Wn)
    sample = np.array(dwt(filtfilt(B, A, channel), 'haar')).flatten()
    return sample

def get_data(redo=True, channels=["Cz"], start=0, end=260):
    '''Helper function for parse_data.

    Args:
        redo: Boolean flag for whether data should be regenerated from scratch,
            or just loaded from disk.
        channels: The EEG data channels to use.
        duration: The number of time-intervals, per sample, to use.
    '''

    print 'Getting data'

    if redo == False: 
        return pd.read_csv('train.csv'), pd.read_csv('test.csv')

    print '========loading train data========'
    train = parse_data("train", channels, start, end, 5440)
    print '========loading test data========'
    test = parse_data("test", channels, start, end, 3400)

    return train, test


def predictions_to_file(algo, train, train_labels, redo, channels, start, end):
    test = parse_data(redo, "test", channels, start, end, 3400)
    algo.fit(train, train_labels)
    preds = algo.predict_proba(test) # why
    preds = preds[:,1]
    submission['Prediction'] = preds
    submission.to_csv('prediction.csv',index=False)


if __name__ == "__main__":

    sensors = ['EOG', 'FCz', 'O2', 'O1', 'Fz', 'TP7', 'CPz', 'C3', 'C2', 'C1', 'C6', 'C5', 'TP8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Cz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'POz', 'Pz', 'C4', 'P4', 'FT7', 'FT8', 'AF8', 'PO7', 'AF4', 'AF3', 'P2', 'P3', 'P1', 'P6', 'P7', 'T8', 'P5', 'T7', 'P8', 'Fp1', 'Fp2', ' AF7', 'P08']

    start_time = time()

    submission = pd.read_csv('data/SampleSubmission.csv')
    #Good: ['C2', 'CPz', 'FC4', 'C1', 'FC2', 'Cz', 'CP4', 'C4', 'Pz', 'FT8', 'P2', 'CP6', 'CP2', 'P4']

    redo = True
    sensorstouse = sensors #['C4']#['Cz','C4']
    start = 30
    end = 150

    print 'Getting data'

    train = parse_data("train", sensorstouse, start, end, 5440)

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


    split = len(train)/2 # location to split data for cross-validation



    algo.fit(train[:split], train_labels[:split]) # train algorithm
    pred = algo.predict(train[split:]) # predict

    # Print prediction report and scores
    print metrics.classification_report(train_labels[split:], pred) # Classification report
    print "GBM score: " + str(algo.score(train[split:], train_labels[split:]))
    print "AUC score:" + str(score(algo, train[split:], train_labels[split:]))

    # Make predictions and save to file
    #predictions_to_file(algo, train, train_labels, redo, sensorstouse, start, end)

    print "Finished."
    print "Running time: " + str(time() - start_time)


