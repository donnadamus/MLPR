#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 03:23:36 2024

@author: marcodonnarumma
"""

import sklearn
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

def computeLDACovMatrixes(D, labels):
   nclasses = 2
   mu = vcol(D.mean(1))
   
   nc = [D[:, labels == i].shape[1] for i in range(nclasses)]
   muClass = numpy.hstack([ vcol(D[:, labels == i].mean(1)) for i in range(nclasses)])
   
   N = D.shape[1] # total samples
   
   # calculate SB
   
   SBc = numpy.zeros((4,4))
   
   for i in range(nclasses):
       d = numpy.subtract(vcol(muClass[:, i]), mu)
       SBc = numpy.add(SBc, nc[i] * numpy.dot(d, d.T))
   
   SB = 1 / N * SBc
   
   # calculate SW
   
   SWc = 0
   SW = numpy.zeros((4,4))
   
   DC = [numpy.subtract(D[:, labels == i], vcol(muClass[:, i])) for i in range(nclasses)]
    
   for i in range(nclasses):
       SWc = numpy.dot(DC[i], DC[i].T)
       SW = numpy.add(SW, SWc)

   SW = 1 / N * SW
    
   return SB, SW

def LDA(D, labels, m):
    
    SB, SW = computeLDACovMatrixes(D, labels)
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    
    DP = numpy.dot(W.T, D)
   
    return DP

def plot_hist(D, L):

    D0 = D[:, L==0] # false
    D1 = D[:, L==1] # true

    plt.figure()
    plt.hist(D0, bins = 100, density = True, alpha = 0.6, label = 'False')
    plt.hist(D1, bins = 100, density = True, alpha = 0.6, label = 'True')
        
    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.show()

if __name__ == '__main__':

    DIris, LIris = load_iris()
    D = DIris[:, LIris != 0]
    L = LIris[LIris != 0]
    
    # DTR and LTR are model training data and labels
    # DVAL and LVAL are validation data and labels
    
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    DPTR = LDA(DTR, LTR, 1) # LDA training set
    DPVA = LDA(DVAL, LVAL, 1) # LDA validation set
    plot_hist(DPTR, LTR)
    
    