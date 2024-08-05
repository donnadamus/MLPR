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

