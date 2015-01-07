## Adapted from phalaris's GBM benchmark. Thanks phalaris.

import time
import sys
import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble as ens
from sklearn import metrics


def score(classifier, X, y):
    fpr, tpr, thresholds = metrics.roc_curve(y, classifier.predict_proba(X)[:,1], pos_label=1)
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


def parse_data(dirName, subs, channels, duration, samples):
    '''Generates X and y to be fed into our predictor, and saves them to disk.

    Args:
        dirName: The directory name holding the data (e.g., "train", "test")
        subs: The subjects.
        channels: The EEG data channels to use.
        duration: The number of time-intervals, per sample, to use.
    '''
    output = pd.DataFrame(
        columns=
            ['subject','session','feedback_num','start_pos'] +
            reduce(lambda x,y: x+y, [[channel+'_'+s for s in map(str,range(duration+1))] for channel in channels]) +
            ['fft_'+freq for freq in map(str,range(1, 48))],
        index=range(samples)
    )
    counter = 0

    for i in subs: # Subjects
        for j in range(1,6): # Sessions (1 - 5)
            # Load in data
            temp = np.load('data/data_processed/'+dirName+'/Data_S' + i + '_Sess0'  + str(j) + '.npy')
            temp = np.delete(temp, (0), axis=0) # delete first row (column labels)

            # Get the positions of the non-zero FeedBackEvents
            feedback_positions = np.flatnonzero(get_column(temp, "FeedBackEvent"))
            
            feedback_counter = 0 # counter keeping track of feedback_num
            for k in feedback_positions:
                temp_time_data = pd.Series(reduce(lambda x,y: x+y, [get_column(temp, channel)[k:k+duration + 1] for channel in channels]))

                sklearn.preprocessing.scale(temp_time_data, copy=False)

                temp2 = pd.concat([temp_time_data, pd.Series(fft(temp_time_data.values))])


                temp2.index = reduce(lambda x,y: x+y, [[channel+'_'+s for s in map(str,range(duration+1))] for channel in channels]) + ['fft_'+freq for freq in map(str,range(1, 48))]


                output.loc[counter, reduce(lambda x,y: x+y, [[channel+'_' + s for s in map(str,range(duration+1))] for channel in channels]) + ['fft_'+freq for freq in map(str,range(1, 48))]] = temp2
                output.loc[counter,'session'] = j
                output.loc[counter, 'subject'] = i
                output.loc[counter, 'feedback_num'] = feedback_counter
                output.loc[counter, 'start_pos'] = k
                counter +=1
                feedback_counter +=1
        sys.stdout.write(i + " ") # Using sys.stdout so we don't print newlines
        sys.stdout.flush()

    print '' # print new line

    output.to_csv(dirName+'.csv',ignore_index=True)
    return output


def get_data(redo=True, channels=["Cz"], duration=260):
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

    # train_subs = ['02']
    train_subs = ['02','06','07','11','12','13','14','16','17','18','20','21','22','23','24','26']
    test_subs = ['01','03','04','05','08','09','10','15','19','25']

    print '========loading train data========'
    train = parse_data("train", train_subs, channels, duration, 5440)
    print '========loading test data========'
    test = parse_data("test", test_subs, channels, duration, 3400)

    return train, test


def predictions_to_file(algo, train, train_labels, test):
    algo.fit(train, train_labels)

    preds = algo.predict_proba(test) # why
    preds = preds[:,1]
    submission['Prediction'] = preds
    submission.to_csv('prediction.csv',index=False)


if __name__ == "__main__":

    # sensors = ['EOG', 'FCz', 'O2', 'O1', 'Fz', 'TP7', 'CPz', 'C3', 'C2', 'C1', 'C6', 'C5', 'TP8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Cz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'POz', 'Pz', 'C4', 'P4', 'FT7', 'FT8', 'AF8', 'PO7', 'AF4', 'AF3', 'P2', 'P3', 'P1', 'P6', 'P7', 'T8', 'P5', 'T7', 'P8', 'Fp1', 'Fp2', ' AF7', 'P08']

    start_time = time.time()

    submission = pd.read_csv('data/SampleSubmission.csv')
    train, test = get_data(True, ["Cz"], 120)
    train_labels = pd.read_csv('data/TrainLabels.csv').values[:,1].ravel().astype(int)

    print 'Training GBM'
    ###     Algorithms      ###
    #algo = ens.AdaBoostClassifier(n_estimators=500, learning_rate=0.05)
    algo = ens.GradientBoostingClassifier(n_estimators=750, learning_rate=0.05)

    split = len(train)/2 # location to split data for cross-validation
    algo.fit(train[:split], train_labels[:split]) # train algorithm
    pred = algo.predict(train[split:]) # predict

    # Print prediction report and scores
    print metrics.classification_report(train_labels[split:], pred) # Classification report
    print "GBM score: " + str(algo.score(train[split:], train_labels[split:]))
    print "AUC score:" + str(score(algo, train[split:], train_labels[split:]))

    # Make predictions and save to file
    predictions_to_file(algo, train, train_labels, test)

    print "Finished."
    print "Running time: " + str(time.time() - start_time)


