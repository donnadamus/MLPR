#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 00:00:25 2024

@author: marcodonnarumma
"""

import numpy
import matplotlib
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1))

def load(fname):
    DAttrs = [] # values for dataset features
    DClass = [] # class for each row of the dataset

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                binaryClass = int(line.split(',')[-1].strip())
                DAttrs.append(attrs)
                DClass.append(binaryClass)
            except:
                pass

    return numpy.hstack(DAttrs), numpy.array(DClass, dtype=numpy.int32)

def plot_hist(D, L):

    D0 = D[:, L==0] # false
    D1 = D[:, L==1] # true

    hFea = {
        0: 'First feature',
        1: 'Second feature',
        2: 'Third feature',
        3: 'Fourth feature',
        4: 'Fifth feature',
        5: 'Sixth feature'
        }

    for dIdx in range(6):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 100, density = True, alpha = 0.6, label = 'False')
        plt.hist(D1[dIdx, :], bins = 100, density = True, alpha = 0.6, label = 'True')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('hist_feature_%d.pdf' % (dIdx + 1))
    plt.show()

def plot_scatter(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    hFea = {
            0: 'First feature',
            1: 'Second feature',
            2: 'Third feature',
            3: 'Fourth feature',
            4: 'Fifth feature',
            5: 'Sixth feature'
    }

    for dIdx1 in range(6):
        for dIdx2 in range(6):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'False')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'True')
        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('scatter_feature%d_feature%d.pdf' % (dIdx1 + 1, dIdx2 + 1))
        plt.show()


if __name__ == '__main__':
    features, labels = load("trainData.txt")
    
    plot_hist(features, labels)
    plot_scatter(features, labels)
    
    muTrueClass = features[:, labels == 1].mean(1).reshape((features.shape[0], 1))
    muFalseClass = features[:, labels == 0].mean(1).reshape((features.shape[0], 1))
    print('Mean True Class:')
    print(muTrueClass)
    print('Mean False Class:')
    print(muFalseClass)
    print()
    
    #    Finally, we can compute the variance of each feature. The variance corresponds to the diagonal of the
    # covariance matrix, and is the square of the standard deviation. Both the variance and standard deviation
    # represent the dispersion of a feature with respect to the class mean, i.e., larger variance implies that,
    # on average, the squared distance of samples from the dataset mean is larger, whereas a small variance
    # indicates that samples are closer to the dataset mean.

    varTrueClass = features[:, labels == 1].var(1)
    varFalseClass = features[:, labels == 0].var(1)
    print('Variance True class:\n', varTrueClass.reshape((features.shape[0], 1)))
    print('Variance False class:\n', varFalseClass.reshape((features.shape[0], 1)))
    print()
    
    
  
    