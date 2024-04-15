#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 03:23:36 2024

@author: marcodonnarumma
"""

import sklearn.datasets
import numpy
import scipy.linalg
import matplotlib.pyplot as plt

def vcol(v):
    return v.reshape((v.size, 1))
    
def vrow(v):
    return v.reshape((1, v.size))

def load_iris():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def computeLDACovMatrixes(D, labels, numfeatures):
   nclasses = 2
   mu = vcol(D.mean(1))
   
   nc = [D[:, labels == (i+1)].shape[1] for i in range(nclasses)]
   muClass = numpy.hstack([ vcol(D[:, labels == (i+1)].mean(1)) for i in range(nclasses)])
   
   N = D.shape[1] # total samples
   
   # calculate SB
   
   SBc = numpy.zeros((numfeatures,numfeatures))
   
   for i in range(nclasses):
       d = numpy.subtract(vcol(muClass[:, i]), mu)
       SBc = numpy.add(SBc, nc[i] * numpy.dot(d, d.T))
   
   SB = 1 / N * SBc
   
   # calculate SW
   
   SWc = 0
   SW = numpy.zeros((numfeatures,numfeatures))
   
   DC = [numpy.subtract(D[:, labels == (i+1)], vcol(muClass[:, i])) for i in range(nclasses)]
    
   for i in range(nclasses):
       SWc = numpy.dot(DC[i], DC[i].T)
       SW = numpy.add(SW, SWc)

   SW = 1 / N * SW
    
   return SB, SW

def LDA(D, labels, m, numfeatures):
    
    SB, SW = computeLDACovMatrixes(D, labels, numfeatures)
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    
    # return W (which columns are the selected directions)
   
    return W

def PCA(D, DSize, m):
    
    # First I calculate the dataset mean and I center the data
    # (it's always worth centering the data before applying PCA) 
    
    mu = D.mean(1)
    DC = D - vcol(mu)
    
    CovMatrix = 1 / DSize * numpy.dot(DC,DC.T)
    
    # Once we have computed the data covariance matrix, 
    # we need to compute its eigenvectors and eigenvalues in order to find
    # the directions with most variance
    
    s, U = numpy.linalg.eigh(CovMatrix)
    
    # which returns the eigenvalues, sorted from smallest to largest, 
    # and the corresponding eigenvectors (columns of U).
    
    # The m leading eigenvectors can be retrieved from U 
    # (here we also reverse the order of the columns of U so 
    # that the leading eigenvectors are in the first m columns):
        
    P = U[:, ::-1][:, 0:m]
    
    return P

def plot_hist(D, L):

    D0 = D[:, L==1] # false
    D1 = D[:, L==2] # true

    plt.figure()
    plt.hist(D0[0, :], bins = 5, density = True, alpha = 0.6, label = "Versicolor")
    plt.hist(D1[0, :], bins = 5, density = True, alpha = 0.6, label = "Virginica")
    plt.legend()

        
    plt.show()

if __name__ == '__main__':

    DIris, LIris = load_iris()
    D = DIris[:, LIris != 0]
    L = LIris[LIris != 0]
    
    # DTR and LTR are model training data and labels
    # DVAL and LVAL are validation data and labels
    
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    """ 
    # The following code builds an LDA classifier (no PCA)
    
    W = LDA(DTR, LTR, 1) # LDA directions training set
    DPTR = numpy.dot(W.T, DTR)  # D projected on W
    DPVA = numpy.dot(W.T, DVAL) # DVAL projected on W
    plot_hist(DPTR, LTR)
    plot_hist(DPVA, LVAL)
    
    # calculate the threshold that we will use to perform inference
    # based on the mean of the classes after applying LDA on the training set
    
    threshold = (DPTR[0, LTR==1].mean() + DPTR[0, LTR==2].mean()) / 2.0
    
    # we will now predict the labels for our validation set
    
    PREDVALSET = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PREDVALSET[DPVA[0] >= threshold] = 2
    PREDVALSET[DPVA[0] < threshold] = 1
    
    errors = numpy.sum(PREDVALSET != LVAL)
    
    """
    
    """
    
    # The following code uses the first PCA direction
    
    P = PCA(DTR, DTR.shape[1], 1)
    DPTR = numpy.dot(P.T, DTR)
    DPVA = numpy.dot(P.T, DVAL)
    threshold = (DPTR[0, LTR==1].mean() + DPTR[0, LTR==2].mean()) / 2.0
    
    # we will now predict the labels for our validation set
    
    PREDVALSET = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PREDVALSET[DPVA[0] >= threshold] = 2
    PREDVALSET[DPVA[0] < threshold] = 1
    
    errors = numpy.sum(PREDVALSET != LVAL)
    
    # PCA alone seems to do very bad. 30 errors out of 34? Maybe I made mistakes?
    
    """
    
    """
    
    
    # The following code uses PCA + LDA
    
    numfeatures = 2
    
    P = PCA(DTR, DTR.shape[1], numfeatures)
    DTRPCA = numpy.dot(P.T, DTR)
    DVALPCA = numpy.dot(P.T, DVAL)
    W = LDA(DTRPCA, LTR, 1, numfeatures=numfeatures) 
    DTRLDA = numpy.dot(W.T, DTRPCA)  
    DVALLDA = numpy.dot(W.T, DVALPCA) 
    
    threshold = (DTRLDA[0, LTR==1].mean() + DTRLDA[0, LTR==2].mean()) / 2.0
    
    # we will now predict the labels for our validation set
    
    PREDVALSET = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PREDVALSET[DVALLDA[0] >= threshold] = 2
    PREDVALSET[DVALLDA[0] < threshold] = 1
    
    errors = numpy.sum(PREDVALSET != LVAL)
    
    print(errors)
    
    # PCA doesn't seem to improve our predictions
    
    """
    
    
    
    
    
    
    
    

    
    
    
    
    
    