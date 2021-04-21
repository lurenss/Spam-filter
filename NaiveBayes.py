import numpy as np
from random import randrange
from time import time
from math import sqrt
from math import exp
from math import pi

""" Separate the dataset by class label values and returns a dictionary """
def separateDataByClass(dataset):
    # Set up dictionary with the two class label options
    dataByClass = dict()
    for label in (0, 1):
        dataByClass[label] = list()

    # For each email in the dataset
    for i in range(len(dataset)):
        # Extract the email's complete feature list and class label
        featureList = dataset[i]
        classLabel = featureList[-1]

        # Clean data to remove features 55-57 since they can be ignored
        featureList = featureList[0:54]
        featureList = np.append(featureList, classLabel)

        # Append feature list
        dataByClass[classLabel].append(featureList)

    return dataByClass

""" Calculate the mean of a list of values  """
def mean(values):
    return sum(values) / float(len(values))

""" Calculate the standard deviation of a list of values """
def stdev(values):
    average = mean(values)
    variance = sum([(x - average) ** 2 for x in values]) / float(len(values) - 1) + 1e-128 # add small constant to avoid dividing by zero
    return sqrt(variance)

""" Summarizes the statistics for each feature in a dataset.
    Includes mean, standard deviation, and count. """
def summarizeDataByFeature(dataset):
    summaries = [(mean(feature), stdev(feature), len(feature)) for feature in zip(*dataset)]
    del (summaries[-1]) # remove the class label feature before returning, since we don't need calculations on this
    return summaries

""" Calculates statistics for features, given a class.
    Separates the dataset by class then summarizes statistics for each feature.
    Includes mean, standard deviation, and count. """
def summarizeDataByClass(dataset):
    dataByClass = separateDataByClass(dataset)
    summaries = dict()

    for class_value, emails in dataByClass.items():
        summaries[class_value] = summarizeDataByFeature(emails)
        del summaries[class_value][-1]  # Remove the class label features since we do not need these

    return summaries

""" Calculates P(X) for a feature X using the Gaussian probability distribution function.
    Since we assume that all features have a Gausssian distribution, we can
    calculate that probability of a given feature X using the mean and stdev. """
def calculateGaussianProbabilityForFeature(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

""" Calculates the probability that an email belongs to each class
    Returns a dictionary of probabilities with one entry for each class label
"""
def calculateClassProbabilities(dataSummarizedByClass, email):
    # Calculate total number of emails in the dataset
    total_emails = sum([dataSummarizedByClass[label][0][2] for label in dataSummarizedByClass])

    # Set of the dictionary of probabilities
    probabilities = dict()

    # For each class in the dataset
    for class_value, class_summaries in dataSummarizedByClass.items():
        # Calculate the probability of a given class, i.e. P(Y = class_value)
        probabilities[class_value] = dataSummarizedByClass[class_value][0][2]/float(total_emails)

        # For each feature in the email
        for i in range(len(class_summaries)):
            # Calculate teh feature's Gaussian probability using the feature's statistics for the given class
            # Accumulate the probabilities by doing multiplication by each feature's probability
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= calculateGaussianProbabilityForFeature(email[i], mean, stdev)

    return probabilities

""" Returns the number of emails per segment, given k folds """
def getFoldSegmentSize(dataset, folds):
    # Figure out the size of each segment if splitting the data in k folds
    return int(len(dataset) / folds)

""" Split the dataset into k folds for cross-validation
    Returns a list of the data split into k segments. """
def splitDataIntoFolds(dataset, folds, segmentSize):
    splitData = list()
    dataset_copy = list(dataset)

    # Split data into k folds and add to the new split list
    for i in range(folds):
        fold = list()
        while len(fold) < segmentSize:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        splitData.append(fold)

    return splitData

""" Calculate accuracy percentage for the class predictions """
def calculateAccuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

""" Calculate the variance for the class predictions """
def calculateVariance(actual, predicted):
    variances = list()
    for i in range(len(actual)):
        variance = sum([(predicted[i] - actual[i]) ** 2]) / float(len(predicted) - 1) + 1e-128 # add small constant to avoid dividing by zero
        variances.append(variance)
    return mean(variances)

""" Evaluate a classification model using k-fold cross-validation """
def evaluateModel(dataset, model, k_folds, *args):
    # Find fold size given the dataset and number of folds
    foldSize = getFoldSegmentSize(dataset, k_folds)

    # Split data into k folds
    splitData = splitDataIntoFolds(dataset, k_folds, foldSize)
    scores = list()
    variances = list()

    # For each fold
    foldNumber = 0
    for fold in splitData:
        # Create the training dataset
        train_set = list(splitData)
        del train_set[foldNumber]
        train_set = sum(train_set, [])
        foldNumber = foldNumber + 1

        # Create the testing dataset
        test_set = list()
        for email in fold:
            email_copy = list(email)
            test_set.append(email_copy)
            email_copy[-1] = None

        # Get predicted class labels for all emails using Naive Bayes
        predicted = model(train_set, test_set, *args)

        # Get actual class labels for all emails
        actual = [email[-1] for email in fold]

        # Calculate the accuracy of the Naive Bayes algorithm
        accuracy = calculateAccuracy(actual, predicted)
        scores.append(accuracy)

        # Calculate the variance of the Naive Bayes algorithm
        variance = calculateVariance(actual, predicted)
        variances.append(variance)

    return [scores, variances]

""" Predict the class label for an email by choosing the class label with the largest probability """
def predictClass(summaries, email):
    mostProbableClass = None
    highestProbability = -1

    # Get the probability for each class for the email
    probabilities = calculateClassProbabilities(summaries, email)

    # Find the class label with the highest probability
    for class_value, probability in probabilities.items():
        if mostProbableClass is None or probability > highestProbability:
            highestProbability = probability
            mostProbableClass = class_value

    return mostProbableClass

""" The main Naive Bayes classification model algorithm """
def naiveBayesAlgorithm(trainingSet, testingSet):
    summarize = summarizeDataByClass(trainingSet)
    predictions = list()
    for email in testingSet:
        output = predictClass(summarize, email)
        predictions.append(output)
    return(predictions)

""" Runs the Naive Bayes Classifier """
def run(dataset, k_folds):
    start = time()
    minAccuracy = 100
    maxAccuracy = 0

    # Evaluate Naive Bayes algorithm
    results = evaluateModel(dataset, naiveBayesAlgorithm, k_folds)
    scores = results[0]
    variances = results[1]

    # Print out accuracy stats for each cross-validation fold, and a mean value at the end
    print("\n>>>>> NAIVE BAYES <<<<<")
    print("--------------------------")

    for i in range(0, k_folds):
        # Set min and max accuracy
        if (scores[i] > maxAccuracy):
                maxAccuracy = scores[i]
        if (scores[i] < minAccuracy):
                minAccuracy = scores[i]

    print("Accuracy (minimum): %.3f%%" % minAccuracy)
    print("Accuracy (maximum): %.3f%%" % maxAccuracy)
    print("Accuracy (mean): %.3f%%" % mean(scores))
    print("Variance: " + str(mean(variances)))

    end = time()
    print("\nTime elapsed: {}".format(end - start))
    print("--------------------------\n")