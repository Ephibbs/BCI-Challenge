## Adapted from phalaris's GBM benchmark. Thanks phalaris.

from __future__ import division
import time
import numpy as np
import pandas as pd
import sklearn.ensemble as ens


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


train_subs = ['02','06','07','11','12','13','14','16','17','18','20','21','22','23','24','26']
test_subs = ['01','03','04','05','08','09','10','15','19','25']

start_time = time.time()

train_labels = pd.read_csv('data/TrainLabels.csv')
submission = pd.read_csv('data/SampleSubmission.csv')

train = pd.DataFrame(columns=['subject','session','feedback_num','start_pos'] + ['Cz_' + s for s in map(str,range(261))],index=range(5440))

# trainz = labelled_array(5440, ['subject', 'session', 'feedback_num', 'start_pos'] + ['Cz_' + s for s in map(str,range(261))]) #######################


counter = 0
print '========train data========'
data = {}
for i in train_subs:
    for j in range(1,6):
        stime = time.time()
        # Load in data
        # temp = pd.read_csv('train/Data_S' + i + '_Sess0'  + str(j) + '.csv')
        temp = np.load('data/data_processed/train/Data_S' + i + '_Sess0'  + str(j) + '.npy')
        temp = np.delete(temp, (0), axis=0) # delete first row (labels)

        # Get the positions of the non-zero FeedBackEvents
        # fb = temp.query('FeedBackEvent == 1',engine='python')['FeedBackEvent']
        fb = np.flatnonzero(get_column(temp, "FeedBackEvent"))
        
        counter2 = 0
        for k in fb: # replaced "fb.index" with "fb"
            # temp2 = temp.loc[int(k):int(k)+260,'Cz']
            temp2 = pd.Series(get_column(temp, "Cz")[k:k+260 + 1])

            temp2.index = ['Cz_' + s for s in map(str,range(261))]
            train.loc[counter,['Cz_' + s for s in map(str,range(261))]] = temp2
            train.loc[counter,'session'] = j
            train.loc[counter, 'subject'] = i
            train.loc[counter, 'feedback_num'] = counter2
            train.loc[counter, 'start_pos'] = k


            # trainz[counter] = np.concatenate((
            #     [i, j, counter2, k],
            #     get_column(temp, "Cz")[k:k+260 + 1].T
            # ), axis=1)


            counter +=1
            counter2 +=1

        print time.time() - stime

    print 'subject ', i

train.to_csv('train_cz.csv',ignore_index=True)

# np.savetxt("foo.csv", trainz, delimiter=",", header=", ".join(trainz.dtype.names), comments="") ########################


test = pd.DataFrame(columns=['subject','session','feedback_num','start_pos'] + ['Cz_' + s for s in map(str,range(261))],index=range(3400))
print '========test data========'
counter = 0
data = {}
for i in test_subs:
    for j in range(1,6):
        stime = time.time()
        # temp = pd.read_csv('test/Data_S' + i + '_Sess0'  + str(j) + '.csv')
        temp = np.load('data/data_processed/test/Data_S' + i + '_Sess0'  + str(j) + '.npy')
        temp = np.delete(temp, (0), axis=0) # delete first row (labels)

        # Get the positions of the non-zero FeedBackEvents
        # fb = temp.query('FeedBackEvent == 1',engine='python')['FeedBackEvent']     
        fb = np.flatnonzero(get_column(temp, "FeedBackEvent"))

        counter2 = 0
        for k in fb:
            # temp2 = temp.loc[int(k):int(k)+260,'Cz']
            temp2 = pd.Series(get_column(temp, "Cz")[k:k+260 + 1])

            temp2.index = ['Cz_' + s for s in map(str,range(261))]
            test.loc[counter,['Cz_' + s for s in map(str,range(261))]] = temp2
            test.loc[counter,'session'] = j
            test.loc[counter, 'subject'] = i
            test.loc[counter, 'feedback_num'] = counter2
            test.loc[counter, 'start_pos'] = k
            counter +=1  
            counter2 +=1

        print time.time() - stime

    print 'subject ', i  

test.to_csv('test_cz.csv',ignore_index=True)

print 'training GBM'


#Probably better to set max_features to default which is sqrt(n_features) or just delete that part, the current value was left in by accident, and probably is not great
gbm = ens.GradientBoostingClassifier(n_estimators=500,learning_rate=0.05, max_features=0.25)
gbm.fit(train, train_labels.values[:,1].ravel())
preds = gbm.predict_proba(test)
preds = preds[:,1]
submission['Prediction'] = preds
submission.to_csv('prediction.csv',index=False)
print 'Done'
print time.time() - start_time
