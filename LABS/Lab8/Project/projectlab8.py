import numpy as np, numpy
import scipy.special
import bayesRisk
import matplotlib.pyplot as plt



def vcol(v):
    return v.reshape((v.size, 1))
    
def vrow(v):
    return v.reshape((1, v.size))

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

if __name__ == '__main__':

    dataset, labels = load("trainData.txt")

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(dataset, labels)

    lambda_vals = np.logspace(-4, 2, 13) # from 10^-4 to 10^2
    print(lambda_vals)
    effPrior = 0.1             # effective prior for the primary application
    list_minDCF = []
    list_actDCF = []

    for lambda_val in lambda_vals:
        w, b = trainLogRegBinary(DTR, LTR, lambda_val)  # return model parameters
        sVal = numpy.dot(w.T, DVAL) + b # Compute validation scores
        PVAL = (sVal > 0) * 1 # Predict validation labels - sVal > 0 returns a boolean array, multiplying by 1 (integer) we get an integer array with 0's and 1's corresponding to the original True and False values
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print ('Error rate: %.1f' % (err*100))

        # --- Compute validation scores in the form of log likelihood ratios, density over density ---

        # Compute empirical prior
        pEmp = (LTR == 1).sum() / LTR.size

        # Compute LLR-like scores
        sValLLR = sVal - numpy.log(pEmp / (1-pEmp))
        minDCF = bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, effPrior, 1.0, 1.0)
        actDCF = bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, effPrior, 1.0, 1.0)
        print ('minDCF - pT (%f): %.4f' % (effPrior, minDCF))
        print ('actDCF - pT (%f): %.4f' % (effPrior, actDCF))

        list_minDCF.append(minDCF)
        list_actDCF.append(actDCF)
    

    # Plot the Bayes error plot
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_vals, list_actDCF, label='actDCF', color='r')
    plt.plot(lambda_vals, list_minDCF, label='minDCF', color='b')
    plt.xscale('log', base=10)  # Usa scala logaritmica per l'asse x
    plt.ylim([0, 1.1])  # Limiti dell'asse y tra 0 e 1.1
    plt.xlabel('Lambda')
    plt.ylabel('Normalized DCF')
    plt.legend()
    plt.show()

    # Increasing lambda means less overfitting, more generalization
    # But we have a smaller norm, so not as good at separating classes (underfitting can occour)

    



