
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split



""" Runs the Support Vector Machine Classifier """
def run(data, p):
    print(">>>>> SUPPORT VECTOR MACHINE <<<<<")
    #data = load_data(TRAINING_SET)
    np.random.shuffle(data)

    #classification for each email in the dataset
    classes = data[:,57] 
    #features for each email in the dataset
    features = data[:,:54] 

    #aplying tfidf transformation
    features = tfidf(features)


    features_train, features_test, classes_train, classes_test = train_test_split(features, classes, test_size=0.3)

    if(p==1):
        linear_svm(features,classes,features_train,classes_train)
    if(p==2):
        polynomial_svm(features,classes,features_train,classes_train)
    if(p==3):
        rbf_svm(features,classes,features_train,classes_train)
    else:
        #normalization of dataset in order to exploit angle 
        norms = np.sqrt(((features+1e-100)**2).sum(axis=1, keepdims=True))
        features_norm = np.where(norms > 0.0, features / norms, 0.)
        features_train_norm, features_test_norm, classes_train_norm, classes_test_norm = train_test_split(features_norm, classes, test_size=0.3)
        if(p==4):
            linear_angle_svm(features_norm,classes,features_train_norm,classes_train_norm)
        if(p==5):
            polynomial_angle_svm(features_norm,classes,features_train_norm,classes_train_norm)    
        if(p==6):
            rbf_angle_svm(features_norm,classes,features_train_norm,classes_train_norm)
     
    """
    linear_svm(features,classes,features_train,classes_train)
    polynomial_svm(features,classes,features_train,classes_train)
    rbf_svm(features,classes,features_train,classes_train)
    norms = np.sqrt(((features+1e-100)**2).sum(axis=1, keepdims=True))
    features_norm = np.where(norms > 0.0, features / norms, 0.)
    features_train_norm, features_test_norm, classes_train_norm, classes_test_norm = train_test_split(features_norm, classes, test_size=0.3)
    linear_angle_svm(features_norm,classes,features_train_norm,classes_train_norm)
    polynomial_angle_svm(features_norm,classes,features_train_norm,classes_train_norm)
    rbf_angle_svm(features_norm,classes,features_train_norm,classes_train_norm)
    """

#TRAINING_SET = "spambase/spambase.data"

def load_data(f_name):
    fr = open(f_name, "r")
    data = np.loadtxt(fr, delimiter=",")
    return data

#tfidf transformation
def tfidf(features):
    tf = features/100.0
    ndoc = features.shape[0]
    #print(ndoc)
    idf = np.log10(ndoc/(features != 0).sum(0))
    #print(idf)
    return tf*idf
    #return tf*idf


def linear_svm(features,classes,features_train,classes_train):
    clf = SVC(kernel="linear",C=1.0)
    results = cross_val_score(clf, features, classes, cv = 10, n_jobs= -1)
    print("--------------------------")
    print("SVM with linear kernel:\n")
    print("Minimum precision: " + str(results.min()))
    print("Maximum precision: " + str(results.max()))
    print("Mean precison: " + str(results.mean()))
    print("Variance: " + str(results.var()))
    clf_fit = clf.fit(features_train, classes_train)
    print("Number of support vectors for a trained SVM : "+ str(clf_fit.n_support_)) 
    print("--------------------------")

def polynomial_svm(features,classes,features_train,classes_train):
    clf = SVC(kernel="poly",degree=2,C=1.0)
    results = cross_val_score(clf, features, classes, cv = 10, n_jobs= -1)
    print("--------------------------")
    print("SVM with second grade polynomial kernel:\n")
    print("Minimum precision: " + str(results.min()))
    print("Maximum precision: " + str(results.max()))
    print("Mean precison: " + str(results.mean()))
    print("Variance: " + str(results.var()))
    clf_fit = clf.fit(features_train, classes_train)
    print("Number of support vectors for a trained SVM : "+ str(clf_fit.n_support_)) 
    print("--------------------------")

def rbf_svm(features,classes,features_train,classes_train):
    clf = SVC(kernel="rbf",C=1.0)
    results = cross_val_score(clf, features, classes, cv = 10, n_jobs= -1)
    print("--------------------------")
    print("SVM with radial basis function kernel:\n")
    print("Minimum precision: " + str(results.min()))
    print("Maximum precision: " + str(results.max()))
    print("Mean precison: " + str(results.mean()))
    print("Variance: " + str(results.var()))
    clf_fit = clf.fit(features_train, classes_train)
    print("Number of support vectors for a trained SVM : "+ str(clf_fit.n_support_)) 
    print("--------------------------")

def linear_angle_svm(features,classes,features_train,classes_train):
    clf = SVC(kernel="linear",C=1.0)
    results = cross_val_score(clf, features, classes, cv = 10, n_jobs= -1)
    print("--------------------------")
    print("SVM with linear kernel (angle):\n")
    print("Minimum precision: " + str(results.min()))
    print("Maximum precision: " + str(results.max()))
    print("Mean precison: " + str(results.mean()))
    print("Variance: " + str(results.var()))
    clf_fit = clf.fit(features_train, classes_train)
    print("Number of support vectors for a trained SVM : "+ str(clf_fit.n_support_)) 
    print("--------------------------")



def polynomial_angle_svm(features,classes,features_train,classes_train):
    clf = SVC(kernel="poly",degree=2,C=1.0)
    results = cross_val_score(clf, features, classes, cv = 10, n_jobs= -1)
    print("--------------------------")
    print("SVM with second grade polynomial (angle) kernel:\n")
    print("Minimum precision: " + str(results.min()))
    print("Maximum precision: " + str(results.max()))
    print("Mean precison: " + str(results.mean()))
    print("Variance: " + str(results.var()))
    clf_fit = clf.fit(features_train, classes_train)
    print("Number of support vectors for a trained SVM : "+ str(clf_fit.n_support_)) 
    print("--------------------------")


def rbf_angle_svm(features,classes,features_train,classes_train):
    clf = SVC(kernel="rbf",C=1.0)
    results = cross_val_score(clf, features, classes, cv = 10, n_jobs= -1)
    print("--------------------------")
    print("SVM with radial basis function (angle) kernel:\n")
    print("Minimum precision: " + str(results.min()))
    print("Maximum precision: " + str(results.max()))
    print("Mean precison: " + str(results.mean()))
    print("Variance: " + str(results.var()))
    clf_fit = clf.fit(features_train, classes_train)
    print("Number of support vectors for a trained SVM : "+ str(clf_fit.n_support_)) 
    print("--------------------------")
