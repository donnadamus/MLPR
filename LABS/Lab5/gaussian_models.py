import sklearn.datasets
import numpy as np
import scipy.special


def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

# Compute a dictionary of ML parameters for each class
def Gau_MVG_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        hParams[lab] = compute_mu_C(DX)
    return hParams

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_mu_C_Tied(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T)
    return mu, C

# This functions returns the density for an array of samples X
def logpdf_GAU_ND(X, mu, C): # MVG
    Y = []
    for i in range (0, X.shape[1]):
        logy = -(X.shape[0]/2)*np.log(2*np.pi)-(1/2)*(np.linalg.slogdet(C)[1])-(1/2)*((np.dot((X[:, i:i+1]-mu).T, np.linalg.inv(C))).T*(X[:, i:i+1]-mu)).sum(axis=0)
        Y.append(logy)
    return np.array(Y).ravel()

def loglikelyhoodperclass(ml_estimates, X):

    toReturn = list()

    for c in ml_estimates.keys():
        muc = ml_estimates[c][0]
        covarc = ml_estimates[c][1]
        toReturn.append(logpdf_GAU_ND(X, muc, covarc))
    
    return np.array(toReturn)

# Compute per-class log-densities. We assume classes are labeled from 0 to C-1. The parameters of each class are in hParams (for class i, hParams[i] -> (mean, cov))
def compute_log_likelihood_Gau(D, hParams):

    S = np.zeros((len(hParams), D.shape[1]))
    for lab in range(S.shape[0]):
        S[lab, :] = logpdf_GAU_ND(D, hParams[lab][0], hParams[lab][1])
    return S


# compute log-postorior matrix from log-likelihood matrix and prior array
def compute_logPosterior(S_logLikelihood, v_prior):
    SJoint = S_logLikelihood + vcol(np.log(v_prior))
    SMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0))
    SPost = SJoint - SMarginal
    return SPost

# Compute a dictionary of ML parameters for each class - Naive Bayes version of the model
# We compute the full covariance matrix and then extract the diagonal. Efficient implementations would work directly with just the vector of variances (diagonal of the covariance matrix)
def Gau_Naive_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C = compute_mu_C(DX)
        hParams[lab] = (mu, C * np.eye(D.shape[0]))
    return hParams

# Compute a dictionary of ML parameters for each class
def Gau_Tied_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        hParams[lab] = compute_mu_C_Tied(DX)

    C = np.sum([hParams[i][1] for i in labelSet], axis=0)
    C = C / len(L)

    for lab in labelSet:
        muc, _ = hParams[lab]
        hParams[lab] = muc, C

    return hParams


def load_iris():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']




if __name__ == '__main__':
    D, L = load_iris()
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # print(DTR.shape) # shape = (4, 100)
    # print(LTR.shape) # shape = (100, )

    # print(DTE.shape) # shape = (4, 50)
    # print(LTR.shape) # shape = (50, )


    # --------- MVG STANDARD CLASSIFIER ---------

    # the covariances matrixes will not be diagonal because the features are not independent in the iris dataset
    ml_estimates = Gau_MVG_ML_estimates(DTR, LTR)

    """

    # Given the estimated model, we now turn our attention towards inference for a test sample x. As we
    # have seen, the final goal is to compute class posterior probabilities P(c|x). We split the process in three
    # stages. The first step consists in computing, for each test sample, the likelihoods

    # densities per each sample per each class

    S = np.exp(loglikelyhoodperclass(ml_estimates=ml_estimates, X=DTE))

    # joint distribution for samples and classes, where 1/3 is the prior probability

    SJoint = S * 1/3

    # check solution for joint distribution
    SolutionSJoint = np.load('Solution/SJoint_MVG.npy')
    # print(np.abs(SJoint - SolutionSJoint).max())

    # now we compute class posterior probabilities

    # sum densities of all the classes for every sample

    SMarginal = vrow(SJoint.sum(0))

    SPost = SJoint / SMarginal

    # SPost.shape is (3, 50)
    # On the rows the class number, while on the columns the samples

    PVAL = SPost.argmax(0)

    print("MVG - Error rate: %.1f%%" % ((PVAL != LTE).sum() / float(LTE.size) * 100))

    """

    S_logLikelihood = compute_log_likelihood_Gau(DTE, ml_estimates)

    S_logPost = compute_logPosterior(S_logLikelihood, np.ones(3)/3.)
    # print ("Max absolute error w.r.t. pre-computed solution - log-posterior matrix")
    # print (np.abs(S_logPost - np.load('Solution/logPosterior_MVG.npy')).max())
    # Predict labels
    PVAL = S_logPost.argmax(0)
    print("MVG - Error rate: %.1f%%" % ((PVAL != LTE).sum() / float(LTE.size) * 100)) 

    print()

    # --------- NAIVE BAYES ---------

        
    hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR)
    for lab in [0,1,2]:
        print('Naive Bayes Gaussian - Class', lab)
        print(hParams_Naive[lab][0])
        print(hParams_Naive[lab][1])
        print()

    S_logLikelihood = compute_log_likelihood_Gau(DTE, hParams_Naive)
    S_logPost = compute_logPosterior(S_logLikelihood, np.ones(3)/3.)
    PVAL = S_logPost.argmax(0)
    print("Naive Bayes Gaussian - Error rate: %.1f%%" % ((PVAL != LTE).sum() / float(LTE.size) * 100))
        
    print()

    # --------- TIED GAUSSIAN ---------

    hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)

    for lab in [0,1,2]:
        print('Tied Gaussian - Class', lab)
        print(hParams_Tied[lab][0])
        print(hParams_Tied[lab][1])
        print()
        

    S_logLikelihood = compute_log_likelihood_Gau(DTE, hParams_Tied)
    S_logPost = compute_logPosterior(S_logLikelihood, np.ones(3)/3.)
    PVAL = S_logPost.argmax(0)
    print("Tied Gaussian - Error rate: %.1f%%" % ((PVAL != LTE).sum() / float(LTE.size) * 100))

    print()

    # --------- BINARY TASKS ---------

    DIris, LIris = load_iris()


    D = DIris[:, LIris != 0]
    L = LIris[LIris != 0]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
    LLR = logpdf_GAU_ND(DVAL, hParams_MVG[2][0], hParams_MVG[2][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1])

    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 2
    PVAL[LLR < TH] = 1
    print("MVG - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))     

    hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)
    LLR = logpdf_GAU_ND(DVAL, hParams_Tied[2][0], hParams_Tied[2][1]) - logpdf_GAU_ND(DVAL, hParams_Tied[1][0], hParams_Tied[1][1])

    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 2
    PVAL[LLR < TH] = 1
    print("Tied - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))   








    


    