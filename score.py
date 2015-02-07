from predict import *

def score(classifier, X, y, cv=5, label=""):
    output = label
    scores = cross_val_score(classifier, X, y, cv)
    output+="\n"+scores

    fpr, tpr, thresholds = metrics.roc_curve(y, classifier.predict(X), pos_label=1)
    auc = metrics.auc(fpr,tpr)
    output+="\n"+auc

    output+="\nGBM score: " + str(algo.score(train[split:], train_labels[split:]))
    output+="\nAUC score:" + str(score(algo, train[split:], train_labels[split:]))

    output+="\n\n\n\n"

    print output
    f = open("results.txt", "r+")
    f.write(output)
    f.close()