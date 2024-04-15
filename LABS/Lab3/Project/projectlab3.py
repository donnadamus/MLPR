#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 23:05:44 2024

@author: marcodonnarumma
"""

import numpy
import scipy.linalg
import matplotlib.pyplot as plt

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

def vcol(v):
    return v.reshape((v.size, 1))
    
def vrow(v):
    return v.reshape((1, v.size))

def load(fname):
    DAttrs = [] # values for dataset features
    DClass = [] # class for each row of the dataset

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = vcol(numpy.array([float(i) for i in attrs]))
                binaryClass = int(line.split(',')[-1].strip())
                DAttrs.append(attrs)
                DClass.append(binaryClass)
            except:
                pass

    return numpy.hstack(DAttrs), numpy.array(DClass, dtype=numpy.int32)

def computeLDACovMatrixes(D, labels, numfeatures):
   nclasses = 2
   mu = vcol(D.mean(1))
   
   nc = [D[:, labels == i].shape[1] for i in range(nclasses)]
   muClass = numpy.hstack([ vcol(D[:, labels == i].mean(1)) for i in range(nclasses)])
   
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
   
   DC = [numpy.subtract(D[:, labels == i], vcol(muClass[:, i])) for i in range(nclasses)]
    
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

def Projection(directions, dataset):
    return numpy.dot(directions.T, dataset)

def plot_hist(D, L):

    D0 = D[:, L==0] # false
    D1 = D[:, L==1] # true

    plt.figure()
    plt.hist(D0[0, :], bins = 100, density = True, alpha = 0.6, label = "False")
    plt.hist(D1[0, :], bins = 100, density = True, alpha = 0.6, label = "True")
    plt.legend()

        
    plt.show()
    
def plot_histPCADirections(D, L, numdirections):

    D0 = D[:, L==0] # false
    D1 = D[:, L==1] # true

    hFea = {
        0: 'First PCA Direction',
        1: 'Second PCA Direction',
        2: 'Third PCA Direction',
        3: 'Fourth PCA Direction',
        4: 'Fifth PCA Direction',
        5: 'Sixth PCA Direction'
        }

    for dIdx in range(numdirections):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 100, density = True, alpha = 0.6, label = 'False')
        plt.hist(D1[dIdx, :], bins = 100, density = True, alpha = 0.6, label = 'True')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('hist_feature_%d.pdf' % (dIdx + 1))
    plt.show()


if __name__ == '__main__':
    dataset, labels = load("trainData.txt")
    
    """
    
    # --- First question (APPLY PCA) ---
    
    P = PCA(dataset, dataset.shape[1], 6)
    datasetPCA = Projection(P, dataset)
    
    plot_histPCADirections(datasetPCA, labels, 6)
    
    # The first PCA direction provides a greater separation of the classes
    # compared to the other directions (although it's not the goal of PCA)
    
    """
    
    """
    
    # --- Second question (APPLY LDA) ---
    
    W = LDA(dataset, labels, 1, 6)
    datasetLDA = Projection(W, dataset)
    plot_hist(datasetLDA, labels)
    
    # The classes overlap but there's a better separation
    # compared to the histograms of lab2
    
    """
    
    # DTR and LTR are model training data and labels
    # DVAL and LVAL are validation data and labels
    
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(dataset, labels)
    
    """
    
    # --- Third question ---
    
    # The following code builds an LDA classifier (no PCA)
    
    W = LDA(DTR, LTR, 1, 6) # LDA directions training set
    DPTR = Projection(W, DTR)
    DPVA = Projection(W, DVAL)
    plot_hist(DPTR, LTR)
    plot_hist(DPVA, LVAL)
    
    # calculate the threshold that we will use to perform inference
    # based on the mean of the classes after applying LDA on the training set
    
    threshold = (DPTR[0, LTR==0].mean() + DPTR[0, LTR==1].mean()) / 2.0
    
    print("Current threshold (mean): ", threshold)
    
    # we will now predict the labels for our validation set
    
    PREDVALSET = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PREDVALSET[DPVA[0] >= threshold] = 1
    PREDVALSET[DPVA[0] < threshold] = 0
    
    errors = numpy.sum(PREDVALSET != LVAL)
    
    # Error rate is equal to 9.30%
    
    print("Error rate: ", errors / LVAL.size * 100)
    
    # --- Fourth question ---
    
    threshold = (numpy.median(DPTR[0, LTR==0]) + numpy.median(DPTR[0, LTR==1])) / 2.0
    
    print("Current threshold (median): ", threshold)
    
    # we will now predict the labels for our validation set
    
    PREDVALSET = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PREDVALSET[DPVA[0] >= threshold] = 1
    PREDVALSET[DPVA[0] < threshold] = 0
    
    errors = numpy.sum(PREDVALSET != LVAL)
    
    # Error rate is equal to 9.30%
    
    print("Error rate: ", errors / LVAL.size * 100)
    
    """
    
    # --- Fifth question ---
    
    numdirections = 2
    
    P = PCA(DTR, DTR.shape[1], numdirections)
    DTRPCA = Projection(P, DTR)
    DVALPCA = Projection(P, DVAL)
    W = LDA(DTRPCA, LTR, 1, numdirections)
    DTRPCALDA = Projection(W, DTRPCA)
    DVALPCALDA = Projection(W, DVALPCA)
    
    threshold = (DTRPCALDA[0, LTR==0].mean() + DTRPCALDA[0, LTR==1].mean()) / 2.0
    
    print("Number of directions: ", numdirections)
    print("Threshold (PCA+LDA): ", threshold)
    
    # we will now predict the labels for our validation set
    
    PREDVALSET = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PREDVALSET[DVALPCALDA[0] >= threshold] = 1
    PREDVALSET[DVALPCALDA[0] < threshold] = 0
    
    errors = numpy.sum(PREDVALSET != LVAL)
    
    print("Error rate: ", errors / LVAL.size * 100)
    
    # PCA is not beneficial in this case, best error rate is obtained
    # with m = 2 and m = 3, 9.25%
    # Pay attention at the mean of the False class and the mean of 
    # the True class, we want the False class being smaller, 
    # because of how we solved the problem
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    

    

