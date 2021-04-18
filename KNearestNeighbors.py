
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


""" Runs the K-Nearest Neighbors Classifier """


def run(data):
    print(">>>>> K-NEAREST NEIGHBORS <<<<<")
    np.random.shuffle(data)

    #classification for each email in the dataset
    classes = data[:,57] 
    #features for each email in the dataset
    features = data[:,:54] 

    #aplying tfidf transformation
    #features = tfidf(features)


    clf = KNeighborsClassifier(n_neighbors=5, p=2, metric='euclidean')
    results = cross_val_score(clf, features, classes, cv = 10, n_jobs= -1)
    print("--------------------------")
    print("KNN classifier with k=5:\n")
    print("Minimum accuracy: " + str(results.min()))
    print("Maximum accuracy: " + str(results.max()))
    print("Mean accuracy: " + str(results.mean()))
    print("Variance: " + str(results.var()))
    print("--------------------------")
