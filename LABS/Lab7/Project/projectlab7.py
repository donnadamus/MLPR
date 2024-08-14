import numpy as np, numpy
import matplotlib.pyplot as plt
import matplotlib

def vcol(v):
    return v.reshape((v.size, 1))
    
def vrow(v):
    return v.reshape((1, v.size))

def Projection(directions, dataset):
    return np.dot(directions.T, dataset)

def PCA(D, DSize, m):
    
    # First I calculate the dataset mean and I center the data
    # (it's always worth centering the data before applying PCA) 
    
    mu = D.mean(1)
    DC = D - vcol(mu)
    
    CovMatrix = 1 / DSize * np.dot(DC,DC.T)
    
    # Once we have computed the data covariance matrix, 
    # we need to compute its eigenvectors and eigenvalues in order to find
    # the directions with most variance
    
    s, U = np.linalg.eigh(CovMatrix)
    
    # which returns the eigenvalues, sorted from smallest to largest, 
    # and the corresponding eigenvectors (columns of U).
    
    # The m leading eigenvectors can be retrieved from U 
    # (here we also reverse the order of the columns of U so 
    # that the leading eigenvectors are in the first m columns):
        
    P = U[:, ::-1][:, 0:m]
    
    return P

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

# Compute a dictionary of ML parameters for each class
def Gau_MVG_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        hParams[lab] = compute_mu_C(DX)
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


def compute_effective_prior(prior, Cfn, Cfp):
    return (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)


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
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

# Specialized function for binary problems (empirical_Bayes_risk is also called DCF or actDCF)
def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * Cfn * Pfn + (1-prior) * Cfp * Pfp
    if normalize:
        return bayesError / numpy.minimum(prior * Cfn, (1-prior)*Cfp)
    return bayesError

# Assume that classes are labeled 0, 1, 2 ... (nClasses - 1)
def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = numpy.zeros((nClasses, nClasses), dtype=numpy.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M

# Optimal Bayes deicsions for binary tasks with log-likelihood-ratio scores
def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -numpy.log( (prior * Cfn) / ((1 - prior) * Cfp) )
    return numpy.int32(llr > th)

# Compute minDCF (slow version, loop over all thresholds recomputing the costs)
# Practical explanation of how minDCF is computed
def compute_minDCF_binary_slow(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    # llrSorter = numpy.argsort(llr) 
    # llrSorted = llr[llrSorter] # We sort the llrs
    # classLabelsSorted = classLabels[llrSorter] # we sort the labels so that they are aligned to the llrs
    # We can remove this part
    llrSorted = llr # In this function (slow version) sorting is not really necessary, since we re-compute the predictions and confusion matrices everytime
    
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), llrSorted, numpy.array([numpy.inf])])
    dcfMin = None
    dcfTh = None
    for th in thresholds:
        predictedLabels = numpy.int32(llr > th)
        dcf = compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp)
        if dcfMin is None or dcf < dcfMin:
            dcfMin = dcf
            dcfTh = th
    if returnThreshold:
        return dcfMin, dcfTh
    else:
        return dcfMin

if __name__ == '__main__':
    # POSITIVE = GENUINE
    # NEGATIVE = FAKE

    # TODO, the PCA results have to be verified. last part with the plot has to be verified as well

    # ----------- Effective priors -----------

    for prior, Cfn, Cfp in [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)]:

        print()
        print('Prior', prior, '- Cfn', Cfn, '- Cfp', Cfp)

        # compute effective prior

        effective_prior = compute_effective_prior(prior, Cfn, Cfp)

        print('Effective prior: ' , effective_prior)
    
    # ----------------------

    dataset, labels = load("trainData.txt")

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(dataset, labels)

    """

    # ----------- NO PCA APPLIED -----------

    # the NO PCA results have been verified and are correct

    print('\n ----------- NO PCA APPLIED ----------- \n')

    for effective_prior, Cfn, Cfp in [(0.1, 1.0, 1.0), (0.5, 1.0, 1.0), (0.9, 1.0, 1.0)]:
        print('\n\n\n')
        print("Effective prior: ", effective_prior)

        hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
        LLR_mvg = logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[0][0], hParams_MVG[0][1])

        predicted_labels_mvg = compute_optimal_Bayes_binary_llr(LLR_mvg, effective_prior, Cfn, Cfp)
        actual_dcf_mvg = compute_empirical_Bayes_risk_binary(predicted_labels_mvg, LVAL, effective_prior, Cfn, Cfp, normalize=True)
        cm = compute_confusion_matrix(predicted_labels_mvg, LVAL)
        print(cm)
        print("DCF (MVG): " , actual_dcf_mvg)
        minDCF_mvg, minDCFth = compute_minDCF_binary_slow(LLR_mvg, LVAL, effective_prior, Cfn, Cfp, returnThreshold=True)
        print("minDCF (MVG): ", minDCF_mvg)

        loss_mvg = (actual_dcf_mvg - minDCF_mvg) / minDCF_mvg * 100
        print("DCF Loss (MVG): ", loss_mvg)


        hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)
        LLR_tied = logpdf_GAU_ND(DVAL, hParams_Tied[1][0], hParams_Tied[1][1]) - logpdf_GAU_ND(DVAL, hParams_Tied[0][0], hParams_Tied[0][1])

        predicted_labels_tied = compute_optimal_Bayes_binary_llr(LLR_tied, effective_prior, Cfn, Cfp)
        actual_dcf_tied = compute_empirical_Bayes_risk_binary(predicted_labels_tied, LVAL, effective_prior, Cfn, Cfp, normalize=True)
        cm = compute_confusion_matrix(predicted_labels_tied, LVAL)
        print(cm)
        print("DCF (Tied): " , actual_dcf_tied)
        minDCF_tied, minDCFth = compute_minDCF_binary_slow(LLR_tied, LVAL, effective_prior, Cfn, Cfp, returnThreshold=True)
        print("minDCF (Tied): ", minDCF_tied)

        loss_tied = (actual_dcf_tied - minDCF_tied) / minDCF_tied * 100
        print("DCF Loss (Tied): ", loss_tied)

        hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR)
        LLR_naive = logpdf_GAU_ND(DVAL, hParams_Naive[1][0], hParams_Naive[1][1]) - logpdf_GAU_ND(DVAL, hParams_Naive[0][0], hParams_Naive[0][1])

        predicted_labels_naive = compute_optimal_Bayes_binary_llr(LLR_naive, effective_prior, Cfn, Cfp)
        actual_dcf_naive = compute_empirical_Bayes_risk_binary(predicted_labels_naive, LVAL, effective_prior, Cfn, Cfp, normalize=True)
        cm = compute_confusion_matrix(predicted_labels_naive, LVAL)
        print(cm)
        print("DCF (Naive): " , actual_dcf_naive)
        minDCF_naive, minDCFth = compute_minDCF_binary_slow(LLR_naive, LVAL, effective_prior, Cfn, Cfp, returnThreshold=True)
        print("minDCF (MVG): ", minDCF_naive)

        loss_naive = (actual_dcf_naive - minDCF_naive) / minDCF_naive * 100
        print("DCF Loss (Naive): ", loss_naive)

    # ----------- PCA APPLIED -----------

    # TODO this results have to be verified. I get a different result compared to the guy that posted on the group, but it might be his mistake

    print('\n ----------- PCA APPLIED ----------- \n')

    # m (numdirections) varies between 1 and 6

    for numdirections in range(1, 7):
        print("!!!!!! m = %d !!!!!!" % numdirections)
        P = PCA(DTR, DTR.shape[1], numdirections)
        DTR_pca = Projection(P, DTR)
        DVAL_pca = Projection(P, DVAL)

        for effective_prior, Cfn, Cfp in [(0.1, 1.0, 1.0), (0.5, 1.0, 1.0), (0.9, 1.0, 1.0)]:
            print('\n\n\n')
            print("Effective prior: ", effective_prior)

            hParams_MVG = Gau_MVG_ML_estimates(DTR_pca, LTR)
            LLR_mvg = logpdf_GAU_ND(DVAL_pca, hParams_MVG[1][0], hParams_MVG[1][1]) - logpdf_GAU_ND(DVAL_pca, hParams_MVG[0][0], hParams_MVG[0][1])

            predicted_labels_mvg = compute_optimal_Bayes_binary_llr(LLR_mvg, effective_prior, Cfn, Cfp)
            actual_dcf_mvg = compute_empirical_Bayes_risk_binary(predicted_labels_mvg, LVAL, effective_prior, Cfn, Cfp, normalize=True)
            cm = compute_confusion_matrix(predicted_labels_mvg, LVAL)
            print(cm)
            print("DCF (MVG): " , actual_dcf_mvg)
            minDCF_mvg, minDCFth = compute_minDCF_binary_slow(LLR_mvg, LVAL, effective_prior, Cfn, Cfp, returnThreshold=True)
            print("minDCF (MVG): ", minDCF_mvg)

            loss_mvg = (actual_dcf_mvg - minDCF_mvg) / minDCF_mvg * 100
            print("DCF Loss (MVG): ", loss_mvg)


            hParams_Tied = Gau_Tied_ML_estimates(DTR_pca, LTR)
            LLR_tied = logpdf_GAU_ND(DVAL_pca, hParams_Tied[1][0], hParams_Tied[1][1]) - logpdf_GAU_ND(DVAL_pca, hParams_Tied[0][0], hParams_Tied[0][1])

            predicted_labels_tied = compute_optimal_Bayes_binary_llr(LLR_tied, effective_prior, Cfn, Cfp)
            actual_dcf_tied = compute_empirical_Bayes_risk_binary(predicted_labels_tied, LVAL, effective_prior, Cfn, Cfp, normalize=True)
            cm = compute_confusion_matrix(predicted_labels_tied, LVAL)
            print(cm)
            print("DCF (Tied): " , actual_dcf_tied)
            minDCF_tied, minDCFth = compute_minDCF_binary_slow(LLR_tied, LVAL, effective_prior, Cfn, Cfp, returnThreshold=True)
            print("minDCF (Tied): ", minDCF_tied)

            loss_tied = (actual_dcf_tied - minDCF_tied) / minDCF_tied * 100
            print("DCF Loss (Tied): ", loss_tied)

            hParams_Naive = Gau_Naive_ML_estimates(DTR_pca, LTR)
            LLR_naive = logpdf_GAU_ND(DVAL_pca, hParams_Naive[1][0], hParams_Naive[1][1]) - logpdf_GAU_ND(DVAL_pca, hParams_Naive[0][0], hParams_Naive[0][1])

            predicted_labels_naive = compute_optimal_Bayes_binary_llr(LLR_naive, effective_prior, Cfn, Cfp)
            actual_dcf_naive = compute_empirical_Bayes_risk_binary(predicted_labels_naive, LVAL, effective_prior, Cfn, Cfp, normalize=True)
            cm = compute_confusion_matrix(predicted_labels_naive, LVAL)
            print(cm)
            print("DCF (Naive): " , actual_dcf_naive)
            minDCF_naive, minDCFth = compute_minDCF_binary_slow(LLR_naive, LVAL, effective_prior, Cfn, Cfp, returnThreshold=True)
            print("minDCF (MVG): ", minDCF_naive)

            loss_naive = (actual_dcf_naive - minDCF_naive) / minDCF_naive * 100
            print("DCF Loss (Naive): ", loss_naive)
    
    """
    
    # ----------- Bayes error plots -----------

    # TODO check this part also

    m = 1 # pca directions that gave the best results in the previous step

    effPriorLogOdds = numpy.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))   # get the corresponding priors


    P = PCA(DTR, DTR.shape[1], m)
    DTR_pca = Projection(P, DTR)
    DVAL_pca = Projection(P, DVAL)

    actDCF = []
    minDCF = []

    # we run this for 3 times (not optimal, we do it to simplify code)

    # standard mvg

    hParams_MVG = Gau_MVG_ML_estimates(DTR_pca, LTR)
    LLR_mvg = logpdf_GAU_ND(DVAL_pca, hParams_MVG[1][0], hParams_MVG[1][1]) - logpdf_GAU_ND(DVAL_pca, hParams_MVG[0][0], hParams_MVG[0][1])

    for effPrior in effPriors:
        predicted_labels_mvg = compute_optimal_Bayes_binary_llr(LLR_mvg, effPrior, Cfn=1.0, Cfp=1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(predicted_labels_mvg, LVAL, effPrior, 1.0, 1.0))

        # I pass to the function the llr directly. No predictions to be made, we scan every possible score and we use it as a threshold
        minDCF.append(compute_minDCF_binary_slow(LLR_mvg, LVAL, effPrior, 1.0, 1.0))
    
    matplotlib.pyplot.figure(0)
    matplotlib.pyplot.plot(effPriorLogOdds, actDCF, label='actDCF MVG')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='minDCF MVG')
    matplotlib.pyplot.ylim([0, 1.1])

    # mvg tied

    actDCF = []
    minDCF = []

    hParams_Tied = Gau_Tied_ML_estimates(DTR_pca, LTR)
    LLR_tied = logpdf_GAU_ND(DVAL_pca, hParams_Tied[1][0], hParams_Tied[1][1]) - logpdf_GAU_ND(DVAL_pca, hParams_Tied[0][0], hParams_Tied[0][1])

    for effPrior in effPriors:
        predicted_labels_tied = compute_optimal_Bayes_binary_llr(LLR_tied, effPrior, Cfn=1.0, Cfp=1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(predicted_labels_tied, LVAL, effPrior, 1.0, 1.0))

        # I pass to the function the llr directly. No predictions to be made, we scan every possible score and we use it as a threshold
        minDCF.append(compute_minDCF_binary_slow(LLR_tied, LVAL, effPrior, 1.0, 1.0))
    
    matplotlib.pyplot.plot(effPriorLogOdds, actDCF, label='actDCF Tied')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='minDCF Tied')
    matplotlib.pyplot.ylim([0, 1.1])

    # mvg naive

    actDCF = []
    minDCF = []

    hParams_Naive = Gau_Naive_ML_estimates(DTR_pca, LTR)
    LLR_naive = logpdf_GAU_ND(DVAL_pca, hParams_Naive[1][0], hParams_Naive[1][1]) - logpdf_GAU_ND(DVAL_pca, hParams_Naive[0][0], hParams_Naive[0][1])

    for effPrior in effPriors:
        predicted_labels_naive = compute_optimal_Bayes_binary_llr(LLR_naive, effPrior, Cfn=1.0, Cfp=1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(predicted_labels_naive, LVAL, effPrior, 1.0, 1.0))

        # I pass to the function the llr directly. No predictions to be made, we scan every possible score and we use it as a threshold
        minDCF.append(compute_minDCF_binary_slow(LLR_naive, LVAL, effPrior, 1.0, 1.0))
    
    matplotlib.pyplot.plot(effPriorLogOdds, actDCF, label='actDCF Naive')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='minDCF Naive')
    matplotlib.pyplot.ylim([0, 1.1])




    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()







    

    