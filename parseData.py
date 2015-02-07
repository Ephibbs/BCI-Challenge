from predict import *

def parse_data(dirName, channels, start, end, samples):
    '''Generates X and y to be fed into our predictor, and saves them to disk.

    Args:
        dirName: The directory name holding the data (e.g., "train", "test")
        subs: The subjects.
        channels: The EEG data channels to use.
        duration: The number of time-intervals, per sample, to use.
    '''

    column = {'Time':0, 'Fp1':1, 'Fp2':2, ' AF7':3, 'AF3':4, 'AF4':5, 'AF8':6, 'F7':7, 'F5':8, 'F3':9, 'F1':10, 'Fz':11, 'F2':12, 'F4':13, 'F6':14, 'F8':15, 'FT7':16, 'FC5':17, 'FC3':18, 'FC1':19, 'FCz':20, 'FC2':21, 'FC4':22, 'FC6':23, 'FT8':24, 'T7':25, 'C5':26, 'C3':27, 'C1':28, 'Cz':29, 'C2':30, 'C4':31, 'C6':32, 'T8':33, 'TP7':34, 'CP5':35, 'CP3':36, 'CP1':37, 'CPz':38, 'CP2':39, 'CP4':40, 'CP6':41, 'TP8':42, 'P7':43, 'P5':44, 'P3':45, 'P1':46, 'Pz':47, 'P2':48, 'P4':49, 'P6':50, 'P8':51, 'PO7':52, 'POz':53, 'P08':54, 'O1':55, 'O2':56, 'EOG':57, 'FeedBackEvent':58}

    channelNums = [column[channel] for channel in channels]

    print '========loading '+dirName+' data========'

    data = np.load(open("data/"+dirName+"_no_pad.npy", "rb"))

    data = data[:,:,channelNums,30:150]
    data = data.reshape((16*340,120))

    return data

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

def butterAndWavelet(channel):
    N = 2
    Wn = 0.01
    B, A = butter(N, Wn)
    sample = np.array(dwt(filtfilt(B, A, channel), 'haar')).flatten()
    return sample