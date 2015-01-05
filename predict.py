## Adapted from phalaris's GBM benchmark. Thanks phalaris.

from __future__ import division
import time
import numpy as np
import pandas as pd
import sklearn.ensemble as ens
from sklearn import metrics

def score(classifier, X, Y):
    fpr, tpr, thresholds = metrics.roc_curve(Y, classifier.predict(X), pos_label=1)
    auc = metrics.auc(fpr,tpr)
    return auc

def labelled_array(y, column_labels=[]):
    '''Create a structured array (numpy array with labelled columns).

    Args:
        y: number of rows
        column_labels: list of column labels; number of columns is assumed size(column_labels)
    '''
    column_labels = [(label, '<f8') for label in column_labels]
    return np.array([tuple([0 for i in range(len(column_labels))]) for j in range(y)], dtype=column_labels)


def get_column(np_array, label):
    '''Returns the column corresponding to the column label in the array. Specific
    to our EEG data.

    Args:
        np_array: the array
        label: the column label
    '''
    column = {'Time':0, 'Fp1':1, 'Fp2':2, ' AF7':3, 'AF3':4, 'AF4':5, 'AF8':6, 'F7':7, 'F5':8, 'F3':9, 'F1':10, 'Fz':11, 'F2':12, 'F4':13, 'F6':14, 'F8':15, 'FT7':16, 'FC5':17, 'FC3':18, 'FC1':19, 'FCz':20, 'FC2':21, 'FC4':22, 'FC6':23, 'FT8':24, 'T7':25, 'C5':26, 'C3':27, 'C1':28, 'Cz':29, 'C2':30, 'C4':31, 'C6':32, 'T8':33, 'TP7':34, 'CP5':35, 'CP3':36, 'CP1':37, 'CPz':38, 'CP2':39, 'CP4':40, 'CP6':41, 'TP8':42, 'P7':43, 'P5':44, 'P3':45, 'P1':46, 'Pz':47, 'P2':48, 'P4':49, 'P6':50, 'P8':51, 'PO7':52, 'POz':53, 'P08':54, 'O1':55, 'O2':56, 'EOG':57, 'FeedBackEvent':58}

    return np_array[:, column[label]]

def parse_data(dirName, subs, channels, duration):
    #helper function parses data and saves to numpy
    output = pd.DataFrame(columns=['subject','session','feedback_num','start_pos'] + reduce(lambda x,y: x+y, [[channel+'_' + s for s in map(str,range(duration+1))] for channel in channels]),index=range(5440))
    counter = 0
    data = {}
    for i in subs:
        for j in range(1,6):
            stime = time.time()
            # Load in data
            # temp = pd.read_csv('train/Data_S' + i + '_Sess0'  + str(j) + '.csv')
            temp = np.load('data/data_processed/'+dirName+'/Data_S' + i + '_Sess0'  + str(j) + '.npy')
            temp = np.delete(temp, (0), axis=0) # delete first row (labels)

            # Get the positions of the non-zero FeedBackEvents
            # fb = temp.query('FeedBackEvent == 1',engine='python')['FeedBackEvent']
            fb = np.flatnonzero(get_column(temp, "FeedBackEvent"))
            
            counter2 = 0
            for k in fb: # replaced "fb.index" with "fb"
                # temp2 = temp.loc[int(k):int(k)+260,'Cz']
                temp2 = pd.Series(reduce(lambda x,y: x+y, [get_column(temp, channel)[k:k+duration + 1] for channel in channels]))

                temp2.index = reduce(lambda x,y: x+y, [[channel+'_' + s for s in map(str,range(duration+1))] for channel in channels])
                output.loc[counter,reduce(lambda x,y: x+y, [[channel+'_' + s for s in map(str,range(duration+1))] for channel in channels])] = temp2
                output.loc[counter,'session'] = j
                output.loc[counter, 'subject'] = i
                output.loc[counter, 'feedback_num'] = counter2
                output.loc[counter, 'start_pos'] = k
                counter +=1
                counter2 +=1

            print time.time() - stime

        print 'subject ', i

    #output.to_csv(dirName+'.csv',ignore_index=True)
    np.save(dirName+'.csv', output.values.astype(float))

    return output.values.astype(float)


def get_data(redo=1, channels=["Cz"], duration=260):
    #redo: reparse the data
    #channels: channels to use
    #duration: number of timesteps of each channel starting with the feedback event
    if redo==0: 
        return np.load("train.npy"), np.load("test.npy")

    train_subs = ['02','06','07','11','12','13','14','16','17','18','20','21','22','23','24','26']
    test_subs = ['01','03','04','05','08','09','10','15','19','25']

    print '========train data========'
    train = parse_data("train", train_subs, channels, duration)

    print '========test data========'
    test = parse_data("test", test_subs, channels, duration)

    return train, test

def predictions_to_file(algo, train, train_labels, test):
    algo.fit(train, train_labels)
    preds = algo.predict_proba(test)
    preds = preds[:,1]
    submission['Prediction'] = preds
    submission.to_csv('prediction.csv',index=False)

if __name__ == "__main__":
    start_time = time.time()

    submission = pd.read_csv('data/SampleSubmission.csv')

    train, test = get_data(0, ["Cz"], 130)
    train_labels = pd.read_csv('data/TrainLabels.csv').values[:,1].astype(float).ravel()

    print 'training GBM'

    #Probably better to set max_features to default which is sqrt(n_features) or just delete that part, the current value was left in by accident, and probably is not great

    ###     Algorithms      ###
    #algo = ens.AdaBoostClassifier(n_estimators=500, learning_rate=0.05)
    algo = ens.GradientBoostingClassifier(n_estimators=500, learning_rate=0.05)

    split = len(train)/2
    algo.fit(train[:split], train_labels[:split])
    pred = algo.predict(train[split:])
    print(metrics.classification_report(train_labels[split:], pred))
    print algo.score(train[split:], train_labels[split:])
    #print score(algo, train[split:], train_labels[split:])

    #predictions_to_file(algo, train, train_labels, test)
    print 'Done'
    print time.time() - start_time
