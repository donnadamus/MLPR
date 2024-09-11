import gmm
import utils
import gaussian
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    f = open('data/trainData.txt', 'r')
    dataSamples, dataLabels = utils.load(f)

    (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(dataSamples, dataLabels)

    data_c0 = DTR[:, LTR == 0]
    mu_c0 = gaussian.mu_ML(data_c0)
    C_c0 = gaussian.C_ML(data_c0, mu_c0)

    data_c1 = DTR[:, LTR == 1]
    mu_c1 = gaussian.mu_ML(data_c1)
    C_c1 = gaussian.C_ML(data_c1, mu_c1)

    minimum_DCF = np.inf
    minimum_actual_DCF = np.inf
    components_c0 = 0
    components_c1 = 0

    for i in [2, 4, 8, 16, 32]:
        for j in [2, 4, 8, 16, 32]:
            gmm_c0 = gmm.train_GMM_LBG_EM(data_c0, i, covType='full', psiEig=0.01, epsLLAverage=1e-6, lbgAlpha=0.1, verbose=True)
            gmm_c1 = gmm.train_GMM_LBG_EM(data_c1, j, covType='full', psiEig=0.01, epsLLAverage=1e-6, lbgAlpha=0.1, verbose=True)

            ll0 = gmm.logpdf_GMM(DVAL, gmm_c0)
            ll1 = gmm.logpdf_GMM(DVAL, gmm_c1)

            ratio = gaussian.llr(ll1, ll0)
            eff_prior = 0.1

            predictions = evaluation.optimal_bayes_decision(ratio, eff_prior, 1, 1)
            confusion_matrix = evaluation.compute_confusion_matrix(predictions, LVAL)
            p_fn = evaluation.FN_rate(confusion_matrix)
            p_fp = evaluation.FP_rate(confusion_matrix)
            actual_dcf = evaluation.DCF_normalized(p_fn, p_fp, 1, 1, eff_prior)
            min_dcf = evaluation.compute_min_dcf(ratio, LVAL, eff_prior, 1, 1)

            if min_dcf < minimum_DCF:
                minimum_DCF = min_dcf
                minimum_actual_DCF = actual_dcf
                components_c0 = i
                components_c1 = j

    print('Minimum DCF found : ', minimum_DCF)
    print('Minimum actual DCF found : ', minimum_actual_DCF)
    print('Number of components class 0 : ', components_c0, ' Number of components class 1 : ', components_c1)

    gmm_c0 = gmm.train_GMM_LBG_EM(data_c0, 8, covType='diagonal', psiEig=0.01, epsLLAverage=1e-6, lbgAlpha=0.1, verbose=True)
    gmm_c1 = gmm.train_GMM_LBG_EM(data_c1, 32, covType='diagonal', psiEig=0.01, epsLLAverage=1e-6, lbgAlpha=0.1, verbose=True)

    ll0_gmm = gmm.logpdf_GMM(DVAL, gmm_c0)
    ll1_gmm = gmm.logpdf_GMM(DVAL, gmm_c1)

    ratio_gmm = gaussian.llr(ll1_gmm, ll0_gmm)
    list_min_dcf_gmm = []
    list_actual_dcf_gmm = []

    kernelFunc = svm.rbfKernel(np.exp(-1))
    C = 1
    eps = 0
    fScore = svm.train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps)
    ratio_svm = fScore(DVAL)
    predictions_svm = (ratio_svm > 0) * 1
    confusion_matrix_svm = evaluation.compute_confusion_matrix(predictions_svm, LVAL)
    p_fn_svm = evaluation.FN_rate(confusion_matrix_svm)
    p_fp_svm = evaluation.FP_rate(confusion_matrix_svm)

    list_min_dcf_svm = []
    list_actual_dcf_svm = []

    l = 0.0031622776601683794
    empirical_prior1 = LTR[LTR == 1].shape[0] / LTR.shape[0]
    DTR_exp = logisticRegression.expand_features(DTR)
    DVAL_exp = logisticRegression.expand_features(DVAL)
    w, b = logisticRegression.trainLogRegBinary(DTR_exp, LTR, l)
    sVal = np.dot(w.T, DVAL_exp) + b
    ratio_log = sVal - np.log(empirical_prior1 / (1 - empirical_prior1))

    list_min_dcf_log = []
    list_actual_dcf_log = []

    effPriorLogOdds = np.linspace(-4, 4, 21)

    for log_odds in effPriorLogOdds:
        eff_prior = evaluation.compute_effective_prior_from_log_odds(log_odds)

        predictions_gmm = evaluation.optimal_bayes_decision(ratio_gmm, eff_prior, 1, 1)
        confusion_matrix_gmm = evaluation.compute_confusion_matrix(predictions_gmm, LVAL)
        p_fn_gmm = evaluation.FN_rate(confusion_matrix_gmm)
        p_fp_gmm = evaluation.FP_rate(confusion_matrix_gmm)
        actual_dcf_gmm = evaluation.DCF_normalized(p_fn_gmm, p_fp_gmm, 1, 1, eff_prior)
        min_dcf_gmm = evaluation.compute_min_dcf(ratio_gmm, LVAL, eff_prior, 1, 1)
        list_min_dcf_gmm.append(min_dcf_gmm)
        list_actual_dcf_gmm.append(actual_dcf_gmm)

        actual_dcf_svm = evaluation.DCF_normalized(p_fn_svm, p_fp_svm, 1, 1, eff_prior)
        min_dcf_svm = evaluation.compute_min_dcf(ratio_svm, LVAL, eff_prior, 1.0, 1.0)
        list_min_dcf_svm.append(min_dcf_svm)
        list_actual_dcf_svm.append(actual_dcf_svm)

        predictions_log = evaluation.optimal_bayes_decision(ratio_log, eff_prior, 1, 1)
        confusion_matrix_log = evaluation.compute_confusion_matrix(predictions_log, LVAL)
        p_fn_log = evaluation.FN_rate(confusion_matrix_log)
        p_fp_log = evaluation.FP_rate(confusion_matrix_log)
        actual_dcf = evaluation.DCF_normalized(p_fn_log, p_fp_log, 1, 1, eff_prior)
        min_dcf = evaluation.compute_min_dcf(ratio_log, LVAL, eff_prior, 1, 1)
        list_min_dcf_log.append(min_dcf)
        list_actual_dcf_log.append(actual_dcf)

    list_min_dcf_log = np.array(list_min_dcf_log)
    list_actual_dcf_log = np.array(list_actual_dcf_log)
    list_min_dcf_svm = np.array(list_min_dcf_svm)
    list_actual_dcf_svm = np.array(list_actual_dcf_svm)
    list_min_dcf_gmm = np.array(list_min_dcf_gmm)
    list_actual_dcf_gmm = np.array(list_actual_dcf_gmm)

    plt.figure(figsize=(10, 6))
    plt.plot(effPriorLogOdds, list_actual_dcf_log, label='DCF', color='r')
    plt.plot(effPriorLogOdds, list_min_dcf_log, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.xlabel('Effective Prior Log-Odds')
    plt.ylabel('Normalized DCF')
    plt.title('Logistic Regression with Expanded Feature - Bayes Error Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(effPriorLogOdds, list_actual_dcf_svm, label='DCF', color='r')
    plt.plot(effPriorLogOdds, list_min_dcf_svm, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.xlabel('Effective Prior Log-Odds')
    plt.ylabel('Normalized DCF')
    plt.title('SVM with RBF Kernel - Bayes Error Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(effPriorLogOdds, list_actual_dcf_gmm, label='DCF', color='r')
    plt.plot(effPriorLogOdds, list_min_dcf_gmm, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.xlabel('Effective Prior Log-Odds')
    plt.ylabel('Normalized DCF')
    plt.title('Diagonal GMM (8 components class 0 and 32 components class 1)- Bayes Error Plot')
    plt.legend()
    plt.grid(True)
    plt.show()