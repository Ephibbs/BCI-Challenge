import numpy as np
import sys

def parse_data(dirName, zeropad=False):
    '''Generates X and y to be fed into our predictor, and saves them to disk.

    Args:
        dirName: The directory name holding the data (e.g., "train", "test")
        subs: The subjects.
        channels: The EEG data channels to use.
        duration: The number of time-intervals, per sample, to use.
    '''

    print '========loading '+dirName+' data========'

    if dirName == 'train':
        subs = ['02','06','07','11','12','13','14','16','17','18','20','21','22','23','24','26']
    elif dirName == 'test':
        subs = ['01','03','04','05','08','09','10','15','19','25']
    else:
        raise

    start = 0
    end = 600
    duration = end - start
    fftlen = duration/2+1 if duration/2+1 < 48 else 48
    counter = 0
    output = np.array([])
    for i in subs: # Subjects
        sub = np.array([])
        for j in range(1,6): # Sessions (1 - 5)
            #sess = np.array([])
            # Load in data
            temp = np.load('data/data_processed/'+dirName+'/Data_S' + i + '_Sess0'  + str(j) + '.npy')
            # Get the positions of the non-zero FeedBackEvents
            temp = np.delete(temp, (0), axis=0) # delete first row (column labels)
            feedback_positions = np.flatnonzero(temp[:,-1])
            temp = np.delete(temp, (-1), axis=1) #delete last column
            
            feedback_counter = 0 # counter keeping track of feedback_num
            for k in feedback_positions:
                temp_time_data = temp[k+start:k+end].T
                counter +=1
                feedback_counter +=1
                # if len(sess) == 0:
                #     sess = np.array([temp_time_data])
                # else:
                #     sess = np.append(sess, [temp_time_data], axis=0)
            # if len(sess) < 100:
            #         sess = np.append(sess, np.zeros((40, sess.shape[1], sess.shape[2])), axis=0)
                if len(sub) == 0:
                    sub = np.array([temp_time_data])
                else:
                    sub = np.append(sub, [temp_time_data], axis=0)
        print sub.shape
        if len(output) == 0:
            output = np.array([sub])
        else:
            output = np.append(output, [sub], axis=0)
            sys.stdout.write(i + " ") # Using sys.stdout so we don't print newlines
            sys.stdout.flush()

    print '' # print new line
    print output.shape
    np.save(open(dirName+'_no_pad.npy', "wb"), output)

parse_data("train")
parse_data("test")