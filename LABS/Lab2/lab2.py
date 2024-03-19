#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import matplotlib
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1))

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
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)

def plot_hist(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }

    for dIdx in range(4):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Setosa')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Versicolor')
        plt.hist(D2[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Virginica')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()

def plot_scatter(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }

    for dIdx1 in range(4):
        for dIdx2 in range(4):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Setosa')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Versicolor')
            plt.scatter(D2[dIdx1, :], D2[dIdx2, :], label = 'Virginica')
        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        plt.show()


if __name__ == '__main__':

   features, labels = load("iris.csv")
   # plot_hist(features, labels)
   plot_scatter(features, labels)
    
   mu = features.mean(1).reshape((features.shape[0], 1))
   print('Mean:')
   print(mu)
   print()

   DC = features - mu
   
   # plot_hist(DC, labels)
   plot_scatter(DC, labels)
   
   # @ is multiplication between matrixes
   
   C = ((features - mu) @ (features - mu).T) / float(features.shape[1])
   print('Covariance:')
   print(C)
   print()
   
   #    Finally, we can compute the variance of each feature. The variance corresponds to the diagonal of the
   # covariance matrix, and is the square of the standard deviation. Both the variance and standard deviation
   # represent the dispersion of a feature with respect to the class mean, i.e., larger variance implies that,
   # on average, the squared distance of samples from the dataset mean is larger, whereas a small variance
   # indicates that samples are closer to the dataset mean.

   var = features.var(1)
   std = features.std(1)
   print('Variance:', var).reshape((features.shape[0], 1))
   print('Std. dev.:', std).reshape((features.shape[0], 1))
   print()







    
    