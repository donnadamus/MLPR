import numpy
import scipy.special
import sklearn.datasets

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

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

# Optimize the logistic regression loss
def trainLogRegBinary(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()
        # return objective and gradient
        return loss.mean() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    # we select the index 0 to get the parameters w (4 dimensions) and b
    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    print ("Log-reg - lambda = %e - J*(w, b) = %e" % (l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

import bayesRisk # Laboratory 7

if __name__ == '__main__':
    
    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    for lambda_val in [1e-3, 1e-1, 1.0]:
        w, b = trainLogRegBinary(DTR, LTR, lambda_val) # Train model
        sVal = numpy.dot(w.T, DVAL) + b # Compute validation scores
        PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print ('Error rate: %.1f' % (err*100))

        # adapt the model to different priors

        # Compute empirical prior
        pEmp = (LTR == 1).sum() / LTR.size
        # Compute LLR-like scores
        sValLLR = sVal - numpy.log(pEmp / (1-pEmp))
        print ('minDCF - pT = 0.5: %.4f' % bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, 0.5, 1.0, 1.0))
        print ('actDCF - pT = 0.5: %.4f' % bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, 0.5, 1.0, 1.0))
        

        
