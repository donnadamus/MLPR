import utils
import numpy as np

def mu_ML(D):
    mu = D.mean(1)
    return mu.reshape(mu.size, 1)

def C_ML(D, mu):
    DC = D - mu
    return (DC @ DC.T) / float(DC.shape[1])

# X  --> a M × N matrix X of samples
# mu --> numpy array of shape (M, 1)
# C  --> numpy array of shape (M, M) representing the covariance matrix Σ
def logpdf_GAU_ND(X, mu, C):
    # M = n_feature (n_lines)
    M = X.shape[0]

    # A = -(M/2)log(2pi)
    A = -(M/2) * np.log(2*np.pi)

    # log(|Σ|)
    _, logdet = np.linalg.slogdet(C)

    # B = -(1/2)log(|Σ|)
    B = -0.5 * logdet

    C_inv = np.linalg.inv(C)
    diff = X - mu

    # C = -(1/2)*(X-mu).T * (Σ^(-1)) * (X-mu)
    C = -0.5 * np.einsum('ij,ji->i', diff.T, np.dot(C_inv, diff))

    return A+B+C

def gaussain_loglikelihood(XND, m_ML, C_ML):
    return logpdf_GAU_ND(XND, m_ML, C_ML).sum()

# This function computes the threshold for the Gaussian Binary Classifier.
# This part is separated from the log likelihood ratio computation because it is APPLICATION DEPENDENTE,
# in fact it uses the prior probability which is application dependent
# check slides 29-32 pdf name 6-GenerativeLinearQuadratic.pdf
def thresholdLLR(prior1, prior2):
    return -np.log(prior1/prior2)

# this function compute the log-likelihood ratio in binary tasks.
def llr(ll1, ll2):
    return ll1 - ll2

# Binary prediction
def predict(llr, threshold):
    return np.where(llr > threshold, 1, 0)

def accuracy(prediction, labels):
    correct_predictions = np.sum(prediction == labels)
    return correct_predictions/labels.shape[0]

def error_rate(prediction, labels):
    return 1 - accuracy(prediction, labels)

