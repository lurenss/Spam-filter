import numpy as np
from time import time
import SupportVectorMachine as SVM
import NaiveBayes as NB
import KNearestNeighbors as KNN

# Load dataset
filename = "spambase/spambase.data"
file = open(filename, "r")
dataset = np.loadtxt(file, delimiter = ",")

# Set 10-fold cross-validation
k_folds = 10

# Run SVM Classifier on dataset
# start = time()
# SVM.run(dataset, k_folds)
# end = time()
# print("\nTime elapsed: {}\n".format(end - start))

# Run Naive Bayes Classifier on dataset
start = time()
NB.run(dataset, k_folds)
end = time()
print("\nTime elapsed: {}\n".format(end - start))

# Run K-Nearest Neighbors Classifier on dataset
# start = time()
# KNN.run(dataset, k_folds)
# end = time()
# print("\nTime elapsed: {}\n".format(end - start))