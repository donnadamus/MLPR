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

    # Binary task
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


