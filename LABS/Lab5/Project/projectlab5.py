import numpy as np
import matplotlib.pyplot as plt

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

# Compute a dictionary of ML parameters for each class
def Gau_MVG_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        hParams[lab] = compute_mu_C(DX)
    return hParams

# This functions returns the density for an array of samples X
def logpdf_GAU_ND(X, mu, C): # MVG
    Y = []
    for i in range (0, X.shape[1]):
        logy = -(X.shape[0]/2)*np.log(2*np.pi)-(1/2)*(np.linalg.slogdet(C)[1])-(1/2)*((np.dot((X[:, i:i+1]-mu).T, np.linalg.inv(C))).T*(X[:, i:i+1]-mu)).sum(axis=0)
        Y.append(logy)
    return np.array(Y).ravel()

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_mu_C_Tied(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T)
    return mu, C

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

if __name__ == '__main__':

    ##### TODO CHECK RESULTS OF PCA PART OF THIS PROJECT LAB 
    
    dataset, labels = load("trainData.txt")

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(dataset, labels)

    # LDA was giving us about 9.3% error rate

    # ----- MVG -----

    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
    LLR = logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[0][0], hParams_MVG[0][1])

    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 1
    PVAL[LLR < TH] = 0
    print("MVG - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100)) 

    print()

    # ----- Tied -----

    hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)

    LLR = logpdf_GAU_ND(DVAL, hParams_Tied[1][0], hParams_Tied[1][1]) - logpdf_GAU_ND(DVAL, hParams_Tied[0][0], hParams_Tied[0][1])

    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 1
    PVAL[LLR < TH] = 0
    print("Tied Gaussian - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100)) 

    print()

    # ----- Naive Bayes -----

    hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR)

    LLR = logpdf_GAU_ND(DVAL, hParams_Naive[1][0], hParams_Naive[1][1]) - logpdf_GAU_ND(DVAL, hParams_Naive[0][0], hParams_Naive[0][1])

    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 1
    PVAL[LLR < TH] = 0
    print("Naive Bayes - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100)) 
        
    print()


    # ----- NOW ANALYZE COVARIANCE MATRIXES -----


    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)

    C0 = hParams_MVG[0][1]
    C1 = hParams_MVG[1][1]

    Corr0 = C0 / ( vcol(C0.diagonal()**0.5) * vrow(C0.diagonal()**0.5) )
    Corr1 = C1 / ( vcol(C1.diagonal()**0.5) * vrow(C1.diagonal()**0.5) )

    print("Correlation matrix class False: ")
    print(Corr0)

    print("Correlation matrix class True: ")
    print(Corr1)

    # The features are weakly correlated, accordingly with the results whe get from Naive Bayes

    # The assumption about the features having a Gaussian distribution is good for the first 4 features
    # but bad for the last 2

    print("-------- USING ONLY THE FIRST 4 FEATURES --------")

    # Even though the last 2 features are not Gaussian distributed, if we remove them
    # we have worse results
    
    dataset, labels = load("trainData.txt", lastfeature=4)

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(dataset, labels)

    # LDA was giving us about 9.3% error rate

    # ----- MVG -----

    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
    LLR = logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[0][0], hParams_MVG[0][1])

    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 1
    PVAL[LLR < TH] = 0
    print("MVG - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100)) 

    print()

    # ----- Tied -----

    hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)

    LLR = logpdf_GAU_ND(DVAL, hParams_Tied[1][0], hParams_Tied[1][1]) - logpdf_GAU_ND(DVAL, hParams_Tied[0][0], hParams_Tied[0][1])

    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 1
    PVAL[LLR < TH] = 0
    print("Tied Gaussian - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100)) 

    print()

    # ----- Naive Bayes -----

    hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR)

    LLR = logpdf_GAU_ND(DVAL, hParams_Naive[1][0], hParams_Naive[1][1]) - logpdf_GAU_ND(DVAL, hParams_Naive[0][0], hParams_Naive[0][1])

    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 1
    PVAL[LLR < TH] = 0
    print("Naive Bayes - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100)) 
        
    print()

    print("-------- USING FEATURES 1 AND 2 --------")

    # Even though the last 2 features are not Gaussian distributed, if we remove them
    # we have worse results
    
    dataset, labels = load("trainData.txt", firstfeature=0, lastfeature=2)

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(dataset, labels)

    # LDA was giving us about 9.3% error rate

    # ----- MVG -----

    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
    LLR = logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[0][0], hParams_MVG[0][1])

    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 1
    PVAL[LLR < TH] = 0
    print("MVG - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100)) 

    print()

    # ----- Tied -----

    hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)

    LLR = logpdf_GAU_ND(DVAL, hParams_Tied[1][0], hParams_Tied[1][1]) - logpdf_GAU_ND(DVAL, hParams_Tied[0][0], hParams_Tied[0][1])

    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 1
    PVAL[LLR < TH] = 0
    print("Tied Gaussian - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100)) 

    print()

    # ----- Naive Bayes -----

    hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR)

    LLR = logpdf_GAU_ND(DVAL, hParams_Naive[1][0], hParams_Naive[1][1]) - logpdf_GAU_ND(DVAL, hParams_Naive[0][0], hParams_Naive[0][1])

    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 1
    PVAL[LLR < TH] = 0
    print("Naive Bayes - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100)) 
        
    print()

    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)

    C0 = hParams_MVG[0][1]
    C1 = hParams_MVG[1][1]

    Corr0 = C0 / ( vcol(C0.diagonal()**0.5) * vrow(C0.diagonal()**0.5) )
    Corr1 = C1 / ( vcol(C1.diagonal()**0.5) * vrow(C1.diagonal()**0.5) )

    print("Correlation matrix class False: ")
    print(Corr0)

    print("Correlation matrix class True: ")
    print(Corr1)

    print("-------- USING FEATURES 3 AND 4 --------")

    # Even though the last 2 features are not Gaussian distributed, if we remove them
    # we have worse results
    
    dataset, labels = load("trainData.txt", firstfeature=2, lastfeature=4)

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(dataset, labels)

    # LDA was giving us about 9.3% error rate

    # ----- MVG -----

    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
    LLR = logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[0][0], hParams_MVG[0][1])

    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 1
    PVAL[LLR < TH] = 0
    print("MVG - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100)) 

    print()

    # ----- Tied -----

    hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)

    LLR = logpdf_GAU_ND(DVAL, hParams_Tied[1][0], hParams_Tied[1][1]) - logpdf_GAU_ND(DVAL, hParams_Tied[0][0], hParams_Tied[0][1])

    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 1
    PVAL[LLR < TH] = 0
    print("Tied Gaussian - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100)) 

    print()

    # ----- Naive Bayes -----

    hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR)

    LLR = logpdf_GAU_ND(DVAL, hParams_Naive[1][0], hParams_Naive[1][1]) - logpdf_GAU_ND(DVAL, hParams_Naive[0][0], hParams_Naive[0][1])

    PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 1
    PVAL[LLR < TH] = 0
    print("Naive Bayes - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100)) 
        
    print()


    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)

    C0 = hParams_MVG[0][1]
    C1 = hParams_MVG[1][1]

    Corr0 = C0 / ( vcol(C0.diagonal()**0.5) * vrow(C0.diagonal()**0.5) )
    Corr1 = C1 / ( vcol(C1.diagonal()**0.5) * vrow(C1.diagonal()**0.5) )

    print("Correlation matrix class False: ")
    print(Corr0)

    print("Correlation matrix class True: ")
    print(Corr1)

    print("----- PCA -----")

    for numdirections in range(1, 7):

        dataset, labels = load("trainData.txt")

        (DTR, LTR), (DVAL, LVAL) = split_db_2to1(dataset, labels)

        print("!!!!!! NUMBER OF DIRECTIONS %d !!!!!!" % numdirections)
    
        P = PCA(DTR, DTR.shape[1], numdirections)
        DTR = Projection(P, DTR)
        DVAL = Projection(P, DVAL)

        # ----- MVG -----

        hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
        LLR = logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[0][0], hParams_MVG[0][1])

        PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
        TH = 0
        PVAL[LLR >= TH] = 1
        PVAL[LLR < TH] = 0
        print("MVG - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100)) 

        print()

        # ----- Tied -----

        hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)

        LLR = logpdf_GAU_ND(DVAL, hParams_Tied[1][0], hParams_Tied[1][1]) - logpdf_GAU_ND(DVAL, hParams_Tied[0][0], hParams_Tied[0][1])

        PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
        TH = 0
        PVAL[LLR >= TH] = 1
        PVAL[LLR < TH] = 0
        print("Tied Gaussian - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100)) 

        print()

        # ----- Naive Bayes -----

        hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR)

        LLR = logpdf_GAU_ND(DVAL, hParams_Naive[1][0], hParams_Naive[1][1]) - logpdf_GAU_ND(DVAL, hParams_Naive[0][0], hParams_Naive[0][1])

        PVAL = np.zeros(DVAL.shape[1], dtype=np.int32)
        TH = 0
        PVAL[LLR >= TH] = 1
        PVAL[LLR < TH] = 0
        print("Naive Bayes - Error rate: %.1f%%" % ((PVAL != LVAL).sum() / float(LVAL.size) * 100)) 
            
        print()



    