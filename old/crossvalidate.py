import cPickle
import sklearn
import random
import numpy as np

 def crossValidate(algo, data, cvtimes=5, cvsize=5):
     accs = []
     for i in range(cvtimes):
         valIs = set([])
         while len(valIs) < cvsize:
             valIs.add(random.randint(0, len(data)))
         valSet = np.array([data[i] for i in valIs]).flatten()
         trainSet = np.array([data[i] for i in range(len(data)) if i not in valIs])
         trainSet = np.reshape(trainSet, (sum([len(a) for a in trainSet]),len(data[0,0])))
         algo.fit(trainSet[:,:,:-1], trainSet[:,:,-1])
         accs.append(algo.score(valSet[:,:,:-1], valSet[:,:,-1]))
    return sum(accs)/cvtimes
