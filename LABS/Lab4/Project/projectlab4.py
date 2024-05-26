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

def computeLogLikelihood(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()

# muml is the empirical dataset mean, varml is the empirical dataset covariance

def computeMaxLikelihood(X):
    muml = vcol(1/X.shape[1] * X.sum(axis=1))   # axis = 1 somma riga
    varml = 1/X.shape[1] * (np.dot((X - muml),(X - muml).T))
    return muml, varml

def load(fname):
    DAttrs = [] # values for dataset features
    DClass = [] # class for each row of the dataset

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = vcol(np.array([float(i) for i in attrs]))
                binaryClass = int(line.split(',')[-1].strip())
                DAttrs.append(attrs)
                DClass.append(binaryClass)
            except:
                pass

    return np.hstack(DAttrs), np.array(DClass, dtype=np.int32)

if __name__ == '__main__':
    dataset, labels = load("trainData.txt")

    DFalse = dataset[:, labels == 0] # shape (6, 2990)
    DTrue = dataset[:, labels == 1] # shape (6, 3010)

    # let's analyze the false class
    for i in range(0, DFalse.shape[0]):
        featureSelected = DFalse[i:i +1, :] # shape (1, 2990)
        muml, varml = computeMaxLikelihood(featureSelected)
        # plot featureSelected and estimated density
        plt.figure()
        plt.hist(featureSelected.ravel(), bins=100, density=True)
        XPlot = np.linspace(featureSelected.min(), featureSelected.max(), 1000)
        plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), muml, varml)))
        plt.savefig('FalseClass_feature_%d.pdf' % (i + 1))
        plt.show()

    # let's analyze the True class
    for i in range(0, DTrue.shape[0]):
        featureSelected = DTrue[i:i +1, :] # shape (1, 2990)
        muml, varml = computeMaxLikelihood(featureSelected)
        # plot featureSelected and estimated density
        plt.figure()
        plt.hist(featureSelected.ravel(), bins=100, density=True)
        XPlot = np.linspace(featureSelected.min(), featureSelected.max(), 1000)
        plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), muml, varml)))
        plt.savefig('TrueClass_feature_%d.pdf' % (i + 1))
        plt.show()



