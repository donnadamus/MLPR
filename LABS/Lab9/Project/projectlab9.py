import numpy as np, numpy
import scipy.special
import sklearn.datasets
import bayesRisk
import matplotlib.pyplot as plt



def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def load(fname, firstfeature=0, lastfeature=6):
    DAttrs = [] # values for dataset features
    DClass = [] # class for each row of the dataset

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[firstfeature:lastfeature]
                attrs = vcol(np.array([float(i) for i in attrs]))
                binaryClass = int(line.split(',')[-1].strip())
                DAttrs.append(attrs)
                DClass.append(binaryClass)
            except:
                pass

    return np.hstack(DAttrs), np.array(DClass, dtype=np.int32)

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

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']    
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

# Optimize SVM
def train_dual_SVM_linear(DTR, LTR, C, K = 1):
    
    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    DTR_EXT = numpy.vstack([DTR, numpy.ones((1,DTR.shape[1])) * K])
    H = numpy.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR_EXT.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)
    
    # Primal loss
    def primalLoss(w_hat):
        S = (vrow(w_hat) @ DTR_EXT).ravel()
        return 0.5 * numpy.linalg.norm(w_hat)**2 + C * numpy.maximum(0, 1 - ZTR * S).sum()

    # here we convert to primal solution
    # Compute primal solution for extended data matrix
    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    
    # Extract w and b - alternatively, we could construct the extended matrix for the samples to score and use directly v
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K # b must be rescaled in case K != 1, since we want to compute w'x + b * K

    primalLoss, dualLoss = primalLoss(w_hat), -fOpt(alphaStar)[0]
    print ('SVM - C %e - K %e - primal loss %e - dual loss %e - duality gap %e' % (C, K, primalLoss, dualLoss, primalLoss - dualLoss))
    
    return w, b

# We create the kernel function. Since the kernel function may need additional parameters, we create a function that creates on the fly the required kernel function
# The inner function will be able to access the arguments of the outer function
def polyKernel(degree, c):
    
    def polyKernelFunc(D1, D2):
        return (numpy.dot(D1.T, D2) + c) ** degree

    return polyKernelFunc

def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):
        # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * numpy.dot(D1.T, D2)
        return numpy.exp(-gamma * Z)

    return rbfKernelFunc

# kernelFunc: function that computes the kernel matrix from two data matrices
def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps = 1.0):

    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    K = kernelFunc(DTR, DTR) + eps
    H = vcol(ZTR) * vrow(ZTR) * K

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)

    print ('SVM (kernel) - C %e - dual loss %e' % (C, -fOpt(alphaStar)[0]))

    # Function to compute the scores for samples in DTE
    def fScore(DTE):
        
        K = kernelFunc(DTR, DTE) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K
        return H.sum(0)

    return fScore # we directly return the function to score a matrix of test samples

if __name__ == '__main__':
    
    dataset, labels = load("/Users/marcodonnarumma/Desktop/MLPR/MLPR/LABS/Lab9/Project/trainData.txt")

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(dataset, labels)

    C_vals = np.logspace(-5, 0, 11) # from 10^-5 to 10^0

    K = 1.0

    # K is the regularization term for the bias (b)

    effPrior = 0.1

    # C is a regularization parameter that controls the trade-off between achieving a low training
    # error and a low testing error, which affects the generalization of the model

    # If C is high, we can say that we are adding high penalty to missclassifications.
    # The model will try to correctly classify as many training points as possible. As we can imagine, less regularization mean we
    # could have high overfitting.

    # If C is low, we can say that we are not adding much penalty to missclassifications.
    # It allows us to have a model that generalizes better, even if that means having some missclassified point

    """

    list_minDCF = []
    list_actDCF = []

    for C in C_vals:
        w, b = train_dual_SVM_linear(DTR, LTR, C, K)
        SVAL = (vrow(w) @ DVAL + b).ravel()
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        minDCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, effPrior, 1.0, 1.0)
        actDCF = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, effPrior, 1.0, 1.0)
        list_minDCF.append(minDCF)
        list_actDCF.append(actDCF)
        print ('Error rate: %.1f' % (err*100))
        print ('minDCF - pT = 0.1: %.4f' % minDCF)
        print ('actDCF - pT = 0.1: %.4f' % actDCF)
        print ()

    plt.figure()
    # Plot the Bayes error plot
    plt.figure(figsize=(10, 6))
    plt.plot(C_vals, list_actDCF, label='actDCF', color='r')
    plt.plot(C_vals, list_minDCF, label='minDCF', color='b')
    plt.xscale('log', base=10)  # Usa scala logaritmica per l'asse x
    plt.ylim([0, 1.1])  # Limiti dell'asse y tra 0 e 1.1
    plt.xlabel('C')
    plt.ylabel('Normalized DCF')
    plt.legend()
    plt.show()
    
    """

    """

    # ----- now center the data -----

    # Compute the mean of each feature in the training set (DTR)
    mean_DTR = np.mean(DTR, axis=1, keepdims=True)  # Calculate the mean across each feature (row-wise)

    # Center the training set (DTR)
    DTR = DTR - mean_DTR

    # Center the validation set (DVAL) using the same mean calculated from the training set
    DVAL = DVAL - mean_DTR


    list_minDCF = []
    list_actDCF = []

    for C in C_vals:
        w, b = train_dual_SVM_linear(DTR, LTR, C, K)
        SVAL = (vrow(w) @ DVAL + b).ravel()
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        minDCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, effPrior, 1.0, 1.0)
        actDCF = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, effPrior, 1.0, 1.0)
        list_minDCF.append(minDCF)
        list_actDCF.append(actDCF)
        print ('Error rate: %.1f' % (err*100))
        print ('minDCF - pT = 0.1: %.4f' % minDCF)
        print ('actDCF - pT = 0.1: %.4f' % actDCF)
        print ()

    plt.figure()
    # Plot the Bayes error plot
    plt.figure(figsize=(10, 6))
    plt.plot(C_vals, list_actDCF, label='actDCF', color='r')
    plt.plot(C_vals, list_minDCF, label='minDCF', color='b')
    plt.xscale('log', base=10)  # Usa scala logaritmica per l'asse x
    plt.ylim([0, 1.1])  # Limiti dell'asse y tra 0 e 1.1
    plt.xlabel('C')
    plt.ylabel('Normalized DCF')
    plt.legend()
    plt.show()

    # POLYNOMIAL KERNEL



    kernelFunc = polyKernel(2, 1)
    eps = 0
    list_minDCF = []
    list_actDCF = []

    for C in C_vals:
        fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps)
        SVAL = fScore(DVAL)
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        minDCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, effPrior, 1.0, 1.0)
        actDCF = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, effPrior, 1.0, 1.0)
        list_minDCF.append(minDCF)
        list_actDCF.append(actDCF)
        print ('Error rate: %.1f' % (err*100))
        print ('minDCF - pT = 0.1: %.4f' % minDCF)
        print ('actDCF - pT = 0.1: %.4f' % actDCF)
        print ()

    plt.figure()
    # Plot the Bayes error plot
    plt.figure(figsize=(10, 6))
    plt.plot(C_vals, list_actDCF, label='actDCF', color='r')
    plt.plot(C_vals, list_minDCF, label='minDCF', color='b')
    plt.xscale('log', base=10)  # Usa scala logaritmica per l'asse x
    plt.ylim([0, 1.1])  # Limiti dell'asse y tra 0 e 1.1
    plt.xlabel('C')
    plt.ylabel('Normalized DCF')
    plt.legend()
    plt.show()

    """

    # RBF KERNEL

    gammas = [np.exp(-4), np.exp(-3), np.exp(-2), np.exp(-1)]
    C_vals = np.logspace(-3, 2, 11) # from 10^-3 to 10^2
    eps = 1

    # each element represents a value of gamma
    list_minDCF = [[], [], [], []]
    list_actDCF = [[], [], [], []]

    indexDCF = 0

    for gamma in gammas:
        kernelFunc = rbfKernel(gamma)
        print("-------- gamma %f --------" % gamma)
        for C in C_vals:
            fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps)
            SVAL = fScore(DVAL)
            PVAL = (SVAL > 0) * 1
            err = (PVAL != LVAL).sum() / float(LVAL.size)
            minDCF = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, effPrior, 1.0, 1.0)
            actDCF = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, effPrior, 1.0, 1.0)
            list_minDCF[indexDCF].append(minDCF)
            list_actDCF[indexDCF].append(actDCF)
            print ('Error rate: %.1f' % (err*100))
            print ('minDCF - pT = 0.1: %.4f' % minDCF)
            print ('actDCF - pT = 0.1: %.4f' % actDCF)
            print ()
        indexDCF+=1



    plt.figure()
    # Plot the Bayes error plot
    plt.figure(figsize=(10, 6))
    plt.plot(C_vals, list_actDCF[0], label=('actDCF, gamma e^-4'), color='r')
    plt.plot(C_vals, list_minDCF[0], label=('minDCF, gamma e^-4'), color='r')

    plt.plot(C_vals, list_actDCF[1], label=('actDCF, gamma e^-3'), color='b')
    plt.plot(C_vals, list_minDCF[1], label=('minDCF, gamma e^-3'), color='b')

    plt.plot(C_vals, list_actDCF[2], label=('actDCF, gamma e^-2'), color='y')
    plt.plot(C_vals, list_minDCF[2], label=('minDCF, gamma e^-2'), color='y')

    plt.plot(C_vals, list_actDCF[3], label=('actDCF, gamma e^-1'), color='g')
    plt.plot(C_vals, list_minDCF[3], label=('minDCF, gamma e^-1'), color='g')

    plt.xscale('log', base=10)  # Usa scala logaritmica per l'asse x
    plt.ylim([0, 1.1])  # Limiti dell'asse y tra 0 e 1.1
    plt.xlabel('C')
    plt.ylabel('Normalized DCF')
    plt.legend()
    plt.show()








