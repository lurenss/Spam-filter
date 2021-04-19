import numpy as np
from time import time
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

""" Runs the Support Vector Machine Classifier """
def run(dataset):
    print(">>>>> SUPPORT VECTOR MACHINE <<<<<")

    for p in range(1, 7):
        start = time()

        np.random.shuffle(dataset)

        classes = dataset[:, 57] # Classification for each email in the dataset
        features = dataset[:, :54] # Features for each email in the dataset

        features = tfidf(features) # Applying tfidf transformation

        features_train, features_test, classes_train, classes_test = train_test_split(features, classes, test_size=0.3)

        if (p == 1):
            svm("linear", None, "linear", features, classes, features_train, classes_train)
        if (p == 2):
            svm("poly", 2, "second grade polynomial", features, classes, features_train, classes_train)
        if (p == 3):
            svm("rbf", None, "radial basis function", features, classes, features_train, classes_train)
        else:
            # Normalization of dataset in order to exploit angle
            norms = np.sqrt(((features+1e-100)**2).sum(axis=1, keepdims=True))
            features_norm = np.where(norms > 0.0, features / norms, 0.)
            features_train_norm, features_test_norm, classes_train_norm, classes_test_norm = train_test_split(features_norm, classes, test_size=0.3)

            if (p == 4):
                svm("linear", None, "linear (angle)", features_norm, classes, features_train_norm, classes_train_norm)
            if (p == 5):
                svm("poly", 2, "second grade polynomial (angle)", features_norm, classes, features_train_norm, classes_train_norm)
            if (p == 6):
                svm("rbf", None, "radial basis function (angle)", features_norm, classes, features_train_norm, classes_train_norm)

        end = time()
        print("\nTime elapsed: {}".format(end - start))
        print("--------------------------")

""" Performs tfidf transformation """
def tfidf(features):
    tf = features/100.0
    ndoc = features.shape[0]
    idf = np.log10(ndoc/(features != 0).sum(0))
    return tf*idf

def printResults(results, title, numVectors):
    print("--------------------------\nSVM with " + title + " kernel:\n")
    print("Accuracy (minimum): %.3f%%" % (results.min() * 100))
    print("Accuracy (maximum): %.3f%%" % (results.max() * 100))
    print("Accuracy (mean): %.3f%%" % (results.mean() * 100))
    print("Variance: " + str(results.var()))
    print("Number of support vectors for a trained SVM: " + str(numVectors))

def svm(kernel, degree, title, features, classes, features_train, classes_train):
    # If we don't explicitly provide a degree, set this to be the default degree = 3 for the SVC method
    if (degree == None):
        degree = 3

    clf = SVC(kernel=kernel, degree=degree, C=1.0)

    results = cross_val_score(clf, features, classes, cv=10, n_jobs=-1)

    clf_fit = clf.fit(features_train, classes_train)

    printResults(results, title, clf_fit.n_support_)