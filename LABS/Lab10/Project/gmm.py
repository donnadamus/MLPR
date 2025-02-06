import numpy as np
import scipy
import scipy.special

import gaussian
import utils


def logpdf_GMM(X, gmm):
    S = []

    for w, mu, C in gmm:
        logpdf_conditional = gaussian.logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + np.log(w)
        S.append(logpdf_joint)

    S = np.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens

# Designed to perform covariance matrix regularization or smoothing.
# This is often necessary in situations where the covariance matrix  C  might be ill-conditioned or singular,
# which can cause issues in various statistical and machine learning algorithms,
# particularly those that require matrix inversion, like in Gaussian Mixture Models (GMMs)
def smooth_covariance_matrix(C, psi):

    U, s, Vh = np.linalg.svd(C)
    s[s<psi]=psi
    CUpd = U @ (utils.mcol(s) * U.T)
    return CUpd


# X: Data matrix
# gmm: input gmm
# covType: 'Full' | 'Diagonal' | 'Tied'
# psiEig: factor for eignvalue thresholding
#
# return: updated gmm
def train_GMM_EM_Iteration(X, gmm, covType='Full', psiEig=None):
    assert (covType.lower() in ['full', 'diagonal', 'tied'])

    # E-step
    S = []

    for w, mu, C in gmm:
        logpdf_conditional = gaussian.logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + np.log(w)
        S.append(logpdf_joint)

    S = np.vstack(S)  # Compute joint densities f(x_i, c), i=1...n, c=1...G
    logdens = scipy.special.logsumexp(S, axis=0)  # Compute marginal for samples f(x_i)

    # Compute posterior for all clusters - log P(C=c|X=x_i) = log f(x_i, c) - log f(x_i)) - i=1...n, c=1...G
    # Each row for gammaAllComponents corresponds to a Gaussian component
    # Each column corresponds to a sample (similar to the matrix of class posterior probabilities in Lab 5, but here the rows are associated to clusters rather than to classes
    gammaAllComponents = np.exp(S - logdens)

    # M-step
    gmmUpd = []
    for gIdx in range(len(gmm)):
        # Compute statistics:
        gamma = gammaAllComponents[gIdx]  # Extract the responsibilities for component gIdx
        Z = gamma.sum()
        F = utils.mcol((utils.mrow(gamma) * X).sum(1))  # Exploit broadcasting to compute the sum
        S = (utils.mrow(gamma) * X) @ X.T
        muUpd = F / Z
        CUpd = S / Z - muUpd @ muUpd.T
        wUpd = Z / X.shape[1]
        if covType.lower() == 'diagonal':
            CUpd = CUpd * np.eye(X.shape[
                                        0])  # An efficient implementation would store and employ only the diagonal terms, but is out of the scope of this script
        gmmUpd.append((wUpd, muUpd, CUpd))

    if covType.lower() == 'tied':
        CTied = 0
        for w, mu, C in gmmUpd:
            CTied += w * C
        gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]

    if psiEig is not None:
        gmmUpd = [(w, mu, smooth_covariance_matrix(C, psiEig)) for w, mu, C in gmmUpd]

    return gmmUpd

# Train a GMM until the average dela log-likelihood becomes <= epsLLAverage
# X is the data for class C
# input gmm

def train_GMM_EM(X, gmm, covType = 'Full', psiEig = None, epsLLAverage = 1e-6, verbose=True):

    llOld = logpdf_GMM(X, gmm).mean()
    llDelta = None
    if verbose:
        print('GMM - it %3d - average ll %.8e' % (0, llOld))
    it = 1
    while (llDelta is None or llDelta > epsLLAverage):
        gmmUpd = train_GMM_EM_Iteration(X, gmm, covType = covType, psiEig = psiEig)
        llUpd = logpdf_GMM(X, gmmUpd).mean()
        llDelta = llUpd - llOld
        if verbose:
            print('GMM - it %3d - average ll %.8e' % (it, llUpd))
        if llDelta < 0:
            print("ERROR - Decreasing")
        gmm = gmmUpd
        llOld = llUpd
        it = it + 1

    if verbose:
        print('GMM - it %3d - average ll %.8e (eps = %e)' % (it, llUpd, epsLLAverage))
    return gmm


def split_GMM_LBG(gmm, alpha = 0.1, verbose=True):

    gmmOut = []
    if verbose:
        print ('LBG - going from %d to %d components' % (len(gmm), len(gmm)*2))
    for (w, mu, C) in gmm:
        U, s, Vh = np.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut

def compute_mu_C(D):
    mu = utils.mcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

# Train a full model using LBG + EM, starting from a single Gaussian model,
# until we have numComponents components.
# lbgAlpha is the value 'alpha' used for LBG, the otehr parameters are the same as in the EM functions above
def train_GMM_LBG_EM(X, numComponents, covType='Full', psiEig=0.01, epsLLAverage=1e-6, lbgAlpha=0.1, verbose=True):
    """
    Trains a Gaussian Mixture Model (GMM) using the Expectation-Maximization (EM) algorithm
    with Lloyd's Binary Splitting (LBG) initialization.

    Parameters:
    ----------
    X : ndarray (features, samples)
        The input dataset, where each column is a sample, and each row is a feature.

    numComponents : int
        Number of Gaussian components (clusters) in the mixture model.

    covType : str, default='Full'
        Type of covariance matrix:
        - 'full' : Each Gaussian has a full covariance matrix (captures correlations).
        - 'diag' : Each Gaussian has a diagonal covariance matrix (assumes independence).

    psiEig : float, default=0.01
        Regularization parameter added to the covariance matrix eigenvalues to prevent singularity.
        Helps stabilize computations when inverting the covariance matrix.

    epsLLAverage : float, default=1e-6
        Convergence threshold for EM training. If the average log-likelihood improvement
        between iterations is below this value, training stops.

    lbgAlpha : float, default=0.1
        Perturbation factor for the LBG algorithm when splitting Gaussian components.
        Higher values create more separated initial clusters.

    verbose : bool, default=True
        If True, prints debug and training progress messages.

    Returns:
    --------
    gmm : dict or custom GMM object
        The trained GMM model containing means, covariances, and mixture weights.
    """

    mu, C = compute_mu_C(X)

    if covType.lower() == 'diagonal':
        C = C * np.eye(X.shape[0])  # We need an initial diagonal GMM to train a diagonal GMM

    if psiEig is not None:
        gmm = [(1.0, mu, smooth_covariance_matrix(C,
                                                  psiEig))]  # 1-component model - if we impose the eignevalus constraint, we must do it for the initial 1-component GMM as well
    else:
        gmm = [(1.0, mu, C)]  # 1-component model

    while len(gmm) < numComponents:
        # Split the components
        if verbose:
            print('Average ll before LBG: %.8e' % logpdf_GMM(X, gmm).mean())
        gmm = split_GMM_LBG(gmm, lbgAlpha, verbose=verbose)
        if verbose:
            print('Average ll after LBG: %.8e' % logpdf_GMM(X,
                                                            gmm).mean())  # NOTE: just after LBG the ll CAN be lower than before the LBG - LBG does not optimize the ll, it just increases the number of components
        # Run the EM for the new GMM
        gmm = train_GMM_EM(X, gmm, covType=covType, psiEig=psiEig, verbose=verbose, epsLLAverage=epsLLAverage)
    return gmm

