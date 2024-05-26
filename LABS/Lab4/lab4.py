import numpy as np
import matplotlib.pyplot as plt

def vcol(v):
    return v.reshape((v.size, 1))
    
def vrow(v):
    return v.reshape((1, v.size))


# Multivariate Gaussian density

"""
The multivariate Gaussian (or normal) distribution is essentially an extension
of the one-dimensional Gaussian (normal) distribution to more dimensions. 

X contains several samples x

x is a sample which contains features with a certain value

This functions returns the density for an array of samples X

"""
def logpdf_GAU_ND(X, mu, C): # MVG
    Y = []
    for i in range (0, X.shape[1]):
        logy = -(X.shape[0]/2)*np.log(2*np.pi)-(1/2)*(np.linalg.slogdet(C)[1])-(1/2)*((np.dot((X[:, i:i+1]-mu).T, np.linalg.inv(C))).T*(X[:, i:i+1]-mu)).sum(axis=0)
        Y.append(logy)
    return np.array(Y).ravel()

if __name__ == '__main__':

    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    mu = np.ones((1,1)) * 1.0
    C = np.ones((1,1)) * 2.0
    Y = np.exp(logpdf_GAU_ND(vrow(XPlot), mu, C))
    plt.plot(XPlot.ravel(), Y)
    plt.show()

    # check solution for multi samples but one dimensional (one variable only)
    pdfSol = np.load('Solution/llGAU.npy')
    pdfGau = logpdf_GAU_ND(vrow(XPlot), mu, C)
    print(np.abs(pdfSol - pdfGau).max())

    # check solution for multi samples and multi-dimensional
    XND = np.load('Solution/XND.npy')
    mu = np.load('Solution/muND.npy')
    C = np.load('Solution/CND.npy')
    pdfSol = np.load('Solution/llND.npy')
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    print(np.abs(pdfSol - pdfGau).max())
