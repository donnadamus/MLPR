import numpy as np

# Calculate False Negative Rate
def FN_rate(confusion_matrix):
    FN = confusion_matrix[0, 1]
    TP = confusion_matrix[1, 1]
    return FN/(TP+FN)

# Calculate False Positive Rate
def FP_rate(confusion_matrix):
    FP = confusion_matrix[1, 0]
    TN = confusion_matrix[0, 0]
    return FP/(TN+FP)

# Calculate True Positive Rate (TPR)
def TP_rate(confusion_matrix):
    FN = confusion_matrix[0, 1]
    TP = confusion_matrix[1, 1]
    return TP / (TP + FN)

# Assume that classes are labeled 0, 1, 2 ... (nClasses - 1)
def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = np.zeros((nClasses, nClasses), dtype=np.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M

# this function compute the optimal Bayes decision
# it receive pi (prior class probability), c_fn (cost false negative) and c_fp (cost false positive)
def optimal_bayes_decision(llr, prior, cost_fn, cost_fp):
    threshold = -np.log((prior*cost_fn)/((1-prior)*cost_fp))
    return (llr>threshold).astype(int)

# DCF_u detection cost function or Empirical Bayes Risk
def DCF(FN_rate, FP_rate, cost_fn, cost_fp, prior):
    return prior*cost_fn*FN_rate + (1-prior)*cost_fp*FP_rate

def DCF_normalized(FN_rate, FP_rate, cost_fn, cost_fp, prior):
    dcf = DCF(FN_rate, FP_rate, cost_fn, cost_fp, prior)
    b_dummy = np.minimum((prior*cost_fn), (1-prior)*cost_fp)
    return dcf/b_dummy

# Function to compute the effective prior
def compute_effective_prior(prior, cost_fn, cost_fp):
    return (prior*cost_fn) / ((prior*cost_fn)+((1-prior)*cost_fp))

# Function to compute the effective prior using log-odds
def compute_effective_prior_from_log_odds(log_odds):
    return 1 / (1 + np.exp(-log_odds))

# Function to compute the minimum normalized DCF for a given effective prior
def compute_min_dcf(llrs, labels, eff_prior, cost_fn, cost_fp):
    thresholds = np.sort(np.unique(llrs))
    dcf_values = []
    # we consider each ratio as a threshold (empirically finding the best one)
    for threshold in thresholds:
        predictions = (llrs > threshold).astype(int)
        cm = compute_confusion_matrix(predictions, labels)
        p_fn = FN_rate(cm)
        p_fp = FP_rate(cm)
        dcf = DCF_normalized(p_fn, p_fp, cost_fn, cost_fp, eff_prior)
        dcf_values.append(dcf)
    return np.min(dcf_values)


def extract_train_val_folds_from_ary(X, idx, KFOLD):
    return np.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]


# Compute minDCF (fast version)
# If we sort the scores, then, as we sweep the scores, we can have that at most one prediction changes everytime. We can then keep a running confusion matrix (or simply the number of false positives and false negatives) that is updated everytime we move the threshold

# Auxiliary function, returns all combinations of Pfp, Pfn corresponding to all possible thresholds
# We do not consider -inf as threshld, since we use as assignment llr > th, so the left-most score corresponds to all samples assigned to class 1 already
def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = np.argsort(llr)
    llrSorted = llr[llrSorter]  # We sort the llrs
    classLabelsSorted = classLabels[llrSorter]  # we sort the labels so that they are aligned to the llrs

    Pfp = []
    Pfn = []

    nTrue = (classLabelsSorted == 1).sum()
    nFalse = (classLabelsSorted == 0).sum()
    nFalseNegative = 0  # With the left-most theshold all samples are assigned to class 1
    nFalsePositive = nFalse

    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)

    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    # The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
    # Pfn.append(1.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    # Pfp.append(0.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    llrSorted = np.concatenate([-np.array([np.inf]), llrSorted])

    # In case of repeated scores, we need to "compact" the Pfn and Pfp arrays (i.e., we need to keep only the value that corresponds to an actual change of the threshold
    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx + 1] != llrSorted[
            idx]:  # We are indeed changing the threshold, or we have reached the end of the array of sorted scores
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])

    return np.array(PfnOut), np.array(PfpOut), np.array(thresholdsOut)  # we return also the corresponding thresholds



# Note: for minDCF llrs can be arbitrary scores, since we are optimizing the threshold
# We can therefore directly pass the logistic regression scores, or the SVM scores
def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):

    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / np.minimum(prior * Cfn, (1-prior)*Cfp) # We exploit broadcasting to compute all DCFs for all thresholds
    idx = np.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]

# Optimal Bayes deicsions for binary tasks with log-likelihood-ratio scores
def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -np.log( (prior * Cfn) / ((1 - prior) * Cfp) )
    return np.int32(llr > th)

# Specialized function for binary problems (empirical_Bayes_risk is also called DCF or actDCF)
def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * Cfn * Pfn + (1-prior) * Cfp * Pfp
    if normalize:
        return bayesError / np.minimum(prior * Cfn, (1-prior)*Cfp)
    return bayesError

# Compute empirical Bayes (DCF or actDCF) risk from llr with optimal Bayes decisions
def compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, classLabels, prior, Cfn, Cfp, normalize=True):
    predictedLabels = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
    return compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=normalize)

compute_actDCF_binary_fast = compute_empirical_Bayes_risk_binary_llr_optimal_decisions # To have a function with a similar name to the minDCF one


def bayesPlot(S, L, left=-4, right=4, npts=21):
    effPriorLogOdds = np.linspace(left, right, npts)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        actDCF.append(compute_actDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
    return effPriorLogOdds, actDCF, minDCF


