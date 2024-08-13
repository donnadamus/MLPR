import numpy
import scipy.special
import matplotlib
import matplotlib.pyplot

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

# compute matrix of posteriors from class-conditional log-likelihoods (each column represents a sample) and prior array
def compute_posteriors(log_clas_conditional_ll, prior_array):
    logJoint = log_clas_conditional_ll + vcol(numpy.log(prior_array))
    logPost = logJoint - scipy.special.logsumexp(logJoint, 0)
    return numpy.exp(logPost)



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


# Specialized function for binary problems (empirical_Bayes_risk is also called DCF or actDCF)
def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * Cfn * Pfn + (1-prior) * Cfp * Pfp
    if normalize:
        return bayesError / numpy.minimum(prior * Cfn, (1-prior)*Cfp)
    return bayesError

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
    
# Auxiliary function, returns all combinations of Pfp, Pfn corresponding to all possible thresholds
# We do not consider -inf as threshld, since we use as assignment llr > th, so the left-most score corresponds to all samples assigned to class 1 already
def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = numpy.argsort(llr)
    llrSorted = llr[llrSorter] # We sort the llrs
    classLabelsSorted = classLabels[llrSorter] # we sort the labels so that they are aligned to the llrs

    Pfp = []
    Pfn = []
    
    nTrue = (classLabelsSorted==1).sum()
    nFalse = (classLabelsSorted==0).sum()
    nFalseNegative = 0 # With the left-most theshold all samples are assigned to class 1
    nFalsePositive = nFalse
    
    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)
    
    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    #The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
    #Pfn.append(1.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    #Pfp.append(0.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    llrSorted = numpy.concatenate([-numpy.array([numpy.inf]), llrSorted])

    # In case of repeated scores, we need to "compact" the Pfn and Pfp arrays (i.e., we need to keep only the value that corresponds to an actual change of the threshold
    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx+1] != llrSorted[idx]: # We are indeed changing the threshold, or we have reached the end of the array of sorted scores
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])
            
    return numpy.array(PfnOut), numpy.array(PfpOut), numpy.array(thresholdsOut) # we return also the corresponding thresholds

if __name__ == '__main__':

    # ----------- Confusion Matrices -----------

    print()
    print("Multiclass - uniform priors and costs - confusion matrix")
    commedia_ll = numpy.load('/Users/marcodonnarumma/Desktop/MLPR/MLPR/LABS/Lab7/Data/commedia_ll.npy')
    commedia_labels = numpy.load('/Users/marcodonnarumma/Desktop/MLPR/MLPR/LABS/Lab7/Data/commedia_labels.npy')

    commedia_posteriors = compute_posteriors(commedia_ll, numpy.ones(3)/3.0)

    # Find the indices of the maximum values along the rows for each column
    labels_posteriors = numpy.argmax(commedia_posteriors, axis=0)

    cm = compute_confusion_matrix(labels_posteriors, commedia_labels)

    print(cm)

    # ----------- Binary task -----------
    print()
    print("-"*40)
    print()
    print("Binary task")
    commedia_llr_binary = numpy.load('/Users/marcodonnarumma/Desktop/MLPR/MLPR/LABS/Lab7/Data/commedia_llr_infpar.npy')
    commedia_labels_binary = numpy.load('/Users/marcodonnarumma/Desktop/MLPR/MLPR/LABS/Lab7/Data/commedia_labels_infpar.npy')

    for prior, Cfn, Cfp in [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]:
        print()
        # prior is referring to class 1 (true)
        print('Prior', prior, '- Cfn', Cfn, '- Cfp', Cfp)
        predicted_labels = compute_optimal_Bayes_binary_llr(commedia_llr_binary, prior, Cfn, Cfp)
        cm = compute_confusion_matrix(predicted_labels, commedia_labels_binary)
        print(cm)

        ebrisk = compute_empirical_Bayes_risk_binary(predicted_labels, commedia_labels_binary, prior, Cfn, Cfp, normalize=False)
        print("Empirical Bayes Risk / DCF (normalize=False): " , ebrisk)

        ebrisk = compute_empirical_Bayes_risk_binary(predicted_labels, commedia_labels_binary, prior, Cfn, Cfp, normalize=True)
        print("Empirical Bayes Risk / DCF (normalize=True): " , ebrisk)

    # ----------- minDCF -----------

    print()
    print("-"*40)
    print()
    print("minDCF")

    for prior, Cfn, Cfp in [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]:
        print()
        # prior is referring to class 1 (true)
        print('Prior', prior, '- Cfn', Cfn, '- Cfp', Cfp)
        minDCF, minDCFth = compute_minDCF_binary_slow(commedia_llr_binary, commedia_labels_binary, prior, Cfn, Cfp, returnThreshold=True)
        print(minDCF)

    # ----------- ROC plot -----------
    
    Pfn, Pfp, _ = compute_Pfn_Pfp_allThresholds_fast(commedia_llr_binary, commedia_labels_binary)
    matplotlib.pyplot.figure(0)
    matplotlib.pyplot.plot(Pfp, 1-Pfn)
    # matplotlib.pyplot.show()

    # ----------- Bayes error plot -----------

    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))   # get the corresponding priors
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        commedia_predictions_binary = compute_optimal_Bayes_binary_llr(commedia_llr_binary, effPrior, 1.0, 1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(commedia_predictions_binary, commedia_labels_binary, effPrior, 1.0, 1.0))

        # I pass to the function the llr directly. No predictions to be made, we scan every possible score and we use it as a threshold
        minDCF.append(compute_minDCF_binary_slow(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0))
    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.plot(effPriorLogOdds, actDCF, label='actDCF eps 0.001', color='r')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='DCF eps 0.001', color='b')
    matplotlib.pyplot.ylim([0, 1.1])
    # matplotlib.pyplot.show()


    # ----------- Bayes error plot COMPARING CLASSIFIERS -----------

    commedia_llr_binary = numpy.load('../Data/commedia_llr_infpar_eps1.npy')
    commedia_labels_binary = numpy.load('../Data/commedia_labels_infpar_eps1.npy')

    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        # Alternatively, we can compute actDCF directly from compute_empirical_Bayes_risk_binary_llr_optimal_decisions(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0)
        commedia_predictions_binary = compute_optimal_Bayes_binary_llr(commedia_llr_binary, effPrior, 1.0, 1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(commedia_predictions_binary, commedia_labels_binary, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_slow(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0))

    matplotlib.pyplot.plot(effPriorLogOdds, actDCF, label='actDCF eps 1.0', color='y')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='DCF eps 1.0', color='c')
    matplotlib.pyplot.ylim([0, 1.1])

    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()


