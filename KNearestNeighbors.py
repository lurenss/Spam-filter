from time import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

""" Runs the K-Nearest Neighbors Classifier """
def run(data, k_nearest):
    print(">>>>> K-NEAREST NEIGHBORS (k=" + str(k_nearest) +") <<<<<")
    start = time()
    np.random.shuffle(data)

    classes = data[:, 57] # Classification for each email in the dataset
    features = data[:, :54] # Features for each email in the dataset

    clf = KNeighborsClassifier(n_neighbors=k_nearest, p=2, metric="euclidean")
    results = cross_val_score(clf, features, classes, cv=10, n_jobs=-1)
    print("--------------------------")
    print("Accuracy (minimum): %.3f%%" % (results.min() * 100))
    print("Accuracy (maximum): %.3f%%" % (results.max() * 100))
    print("Accuracy (mean): %.3f%%" % (results.mean() * 100))
    print("Variance: " + str(results.var()))

    end = time()
    print("\nTime elapsed: {}".format(end - start))
    print("--------------------------\n")