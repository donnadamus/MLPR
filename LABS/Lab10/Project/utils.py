import numpy as np
import math
import matplotlib.pyplot as plt
import scipy


def mcol(vector):
    """Trasforma un vettore 1D in un vettore colonna."""
    return np.array(vector).reshape(-1, 1)

def mrow(vector):
    """Trasforma un vettore 1D in un vettore riga."""
    return np.array(vector).reshape(1, -1)

def load(f):
    v = []
    m = []

    for line in f:
        l = line.rstrip().split(',')
        conversion = [float(i) for i in l[0:6]] #10 dimension
        m.append(conversion)
        v.append(int(l[6]))

    vector = np.array(v)
    matrix = np.array(m)

    matrix = matrix.T

    return matrix, vector

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)


def plot_scatter(m, v):
    m0 = m[:, v == 0]
    m1 = m[:, v == 1]

    for i in range(m.shape[0]):
        for j in range(m.shape[0]):
            if (i == j):
                continue
            else:
                plt.figure()
                plt.xlabel("Feature " + str(i+1))
                plt.ylabel("Feature " + str(j+1))
                plt.scatter(m0[i, :], m0[j, :], color='r')
                plt.scatter(m1[i, :], m1[j, :], color='b')

                plt.legend()
                plt.tight_layout()
            plt.show()

def plot_histogram(m, v):
    for i in range(m.shape[0]):
        plt.figure()
        plt.xlabel('Feature ' + str(i + 1))
        plt.ylabel('Frequency')
        m0 = m[:, v == 0]
        m1 = m[:, v == 1]
        plt.hist(m0[i, :], bins='auto', density=True, edgecolor="black", alpha = 0.5, label="spoofed fingerprint")
        plt.hist(m1[i, :], bins='auto', density=True, edgecolor="black", alpha = 0.5, label="authentic fingerprint")

        plt.legend()
        # Use with non-default font size to keep axis label inside the figure
        plt.tight_layout()
    plt.show()
    return

def PCA(data, m):
    mu = data.mean(1) # su riga
    # we reshape mu to be a column vector, we can then remove it from all columns of D simply with
    centered_data = data - mcol(mu)

    # C = Covariance Matrix
    C = (centered_data @ centered_data.T) / float(centered_data.shape[1])
    U, s, Vh = np.linalg.svd(C) # this function returns the eigenvectors in descending order
    P = U[:, 0:m] # N_features x m --> 6 x m
    # • HOW PROJECTION WORKS :   P.T --> m x 6  (dot)  6 x N [data]   --resulting-->  m x N (N sample with m feature)
    return P

def compute_Sw_Sb(data, L):

    # f = n_features,  N = n_samples
    f, n = data.shape
    # total mean of the dataset
    total_mu = mcol(data.mean(1))
    classes = np.unique(L)

    # Initialization of S_w e S_b
    # S_w and S_b dimensions depends on the number of features
    # n_features * n_features
    S_w = np.zeros((f, f))
    S_b = np.zeros((f, f))

    for cls in classes:
        # Select the samples of the i–th class as
        data_c = data[:, L == cls]
        n_c = data_c.shape[1]

        # Compute mean for the actual class
        mu_c = mcol(data_c.mean(1))

        # Calcola la covarianza intra-classe e aggiorna S_w
        data_c_centered = (data_c - mu_c)
        S_w += data_c_centered @ data_c_centered.T

        # compute mean difference
        mu_diff = mu_c - total_mu
        S_b += n_c * (mu_diff @ mu_diff.T)

    # compute 1/N
    S_b = S_b / n
    S_w = S_w / n

    return S_w, S_b

def compute_Sw(data, L):

    # f = n_features,  n = n_samples
    f, n = data.shape
    # total mean of the dataset
    total_mu = data.mean(1).reshape(-1, 1)

    classes = np.unique(L)

    # Initialization of S_w e S_b
    # S_w and S_b dimensions depends on the number of features
    # n_features * n_features
    S_w = np.zeros((f, f))

    for cls in classes:
        # Select the samples of the i–th class as
        data_c = data[:, L == cls]

        # Compute mean for the actual class
        mu_c = mcol(data_c.mean(1))


        DC_c = (data_c - mu_c)
        S_w += DC_c @ DC_c.T

    S_w = S_w / n
    return S_w

# Remember that we can compute at most n_C-1 discriminant directions, where n_C is the number of classes
def LDA(data, labels, m):
    mu = data.mean(1) # su riga
    # we reshape mu to be a column vector, we can then remove it from all columns of D simply with
    centered_data = data - mcol(mu)
    SW, SB = compute_Sw_Sb(centered_data, labels)

    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]

    UW, _, _ = np.linalg.svd(W)
    P = UW[:, 0:m]
    # • HOW PROJECTION WORKS :   P.T --> m x 6  (dot)  6 x N [data]   --resulting-->  m x N (N sample with m feature)
    return P
