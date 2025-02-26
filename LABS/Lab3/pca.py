#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:22:08 2024

@author: marcodonnarumma
"""

import numpy
import matplotlib
import matplotlib.pyplot as plt

def vcol(v):
    return v.reshape((v.size, 1))
    
def vrow(v):
    return v.reshape((1, v.size))

def load(fname):
    DList = []
    labelsList = []
    hLabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
        }

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = vcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)

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
    
    # Finally, we can apply the projection to a matrix of samples D as:
        
    DP = numpy.dot(P.T, D)
    
    # yes, we found the PCA subspace using the centered dataset, but once 
    # the subspace directions have been found, we use the dataset points
    # (which btw is the one we really are interested in) to plot the data
    
    return DP

def plot_scatter2Dim(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]

    plt.figure()
    plt.xlabel("PC0")
    plt.ylabel("PC1")
    plt.scatter(D0[0, :], D0[1, :], label = 'Setosa')
    plt.scatter(D1[0, :], D1[1, :], label = 'Versicolor')
    plt.scatter(D2[0, :], D2[1, :], label = 'Virginica')
        
    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig('scatter_PCA.pdf')
    plt.show()
    
if __name__ == '__main__':

   D, labels = load("iris.csv")
   
   # here I use labels.size to find the number of samples that I had in my dataset,
   # considering that I had one label for each sample
   
   DPCA = PCA(D, labels.size, 2)
   plot_scatter2Dim(DPCA, labels)
   
   
   
   