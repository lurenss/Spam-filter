import numpy as np
from time import time
import SupportVectorMachine as SVM
import NaiveBayes as NB
import KNearestNeighbors as KNN

# Load dataset
filename = "spambase/spambase.data"
file = open(filename, "r")
dataset = np.loadtxt(file, delimiter = ",")

# Run SVM Classifier on dataset
# Will run through different kernels: linear, polynomial, RBF, linear angular, polynomial angular, RBF angular
SVM.run(dataset)

# Run Naive Bayes Classifier on dataset with 10-fold cross validation
k_folds = 10
NB.run(dataset, k_folds)

# Run K-Nearest Neighbors Classifier on dataset with k=5
k_nearest = 5
KNN.run(dataset, k_nearest)