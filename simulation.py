"""
This script replicates the work by Ng et al. IPMI 2015

We start with a simpler setting, based on Lasso.

Author: Bertrand Thirion, 2015 
"""

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, MultiTaskLassoCV, lasso_path
from sklearn import metrics
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import precision_recall_fscore_support


def SMR(X, y):
    model = LassoCV().fit(X, y)
    return np.abs(model.coef_) > 0


def accuracy_(support, candidate):
    true_positives = support * candidate
    false_positives = (support == 0) * candidate
    fpr = 1. * false_positives.sum() / (1. - support).sum()
    if support.sum() == 0:
        tpr = 0
    else:
        tpr = 1. * true_positives.sum() / support.sum()
    return fpr, tpr


def lasso_coefs(X, y):
    _, coefs, _ = lasso_path(X, y)
    return coefs.T


def precision_recall(true, estimated):
    """ Compute precision and recall from list of estimators"""
    precision, recall = [], []
    p_old, r_old = None, None
    for estimated_ in estimated:
        if not estimated_.any():
            continue
        p, r, _, _ = precision_recall_fscore_support(
            true, np.abs(estimated_) > 0, average='micro')
        if p_old is not None:
            if  p == p_old and r == r_old:
                continue
            precision.append(p)
            recall.append(r)

        p_old = p
        r_old = r
    return precision, recall


def accuracy(support, coef):
    return accuracy_(support, np.abs(coef) > 0)


def stability_selection(X, y, n_bootstraps=100, pi=.75):
    n_samples, n_features = X.shape
    alpha_max = np.max(np.dot(y, X)) / n_samples
    alpha = .1 * alpha_max
    stability = np.zeros(n_features)
    for b in range(n_bootstraps):
        X_ = .5 * X * (1 + (np.random.rand(n_features) > 0.5))
        mask = np.random.rand(n_samples) > 0.5
        X_, y_ = X_[mask], y[mask]
        coef = Lasso(alpha=alpha).fit(X_, y_).coef_
        stability += np.abs(coef) > 0

    if pi is None:
        return stability
    else:
        return stability > pi * n_bootstraps


class BootstrapLasso(Lasso):

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 precompute=False, copy_X=True, max_iter=1000, tol=0.0001,
                 warm_start=False, positive=False, random_state=None,
                 selection='cyclic', n_bootstraps=100):
        super(BootstrapLasso, self).__init__(
            alpha, fit_intercept, normalize, precompute,
            copy_X, max_iter, tol, warm_start,
            positive, random_state, selection)
        self.n_bootstraps = n_bootstraps

    def fit(self, X, y):
        """ returns Studentized coefficients"""
        clf = Lasso(
            alpha=self.alpha, fit_intercept=self.fit_intercept,
            normalize=self.normalize, precompute=self.precompute,
            copy_X=self.copy_X, max_iter=self.max_iter, tol=self.tol,
            warm_start=self.warm_start, positive=self.positive,
            random_state=self.random_state, selection=self.selection)
        n_samples, n_features = X.shape
        coefs = np.empty((n_bootstraps, n_features), np.float)
        for b in range(self.n_bootstraps):
            samples = np.random.randint(0, n_samples, n_samples)
            random_weight = .5 * np.random.randint(1, 3, n_features)
            y_, X_ = y[samples], X[samples] * random_weight
            coefs[b] = clf.fit(X_, y_).coef_

        self.coef_ = coefs.mean(0) / np.maximum(1.e-12, coefs.std(0))
        return self


def permutation_testing(X, y, n_perm=100000, pval=.05):
    """ Permutation testing"""
    n_samples, n_features = X.shape
    alpha_max = np.max(np.dot(y, X)) / n_samples
    alpha = .1 * alpha_max
    clf = Lasso(alpha=alpha)
    coef = clf.fit(X, y).coef_
    pcoef = np.zeros(n_perm)
    plop = np.zeros((n_perm, n_features))
    for perm in range(n_perm):
        y_ = y * (2 * (np.random.randint(0, 2, n_samples) - .5))
        # alpha_ = .1 * np.max(np.dot(y_, X)) / n_samples
        # clf.alpha = alpha_
        coef_ = clf.fit(X, y_).coef_
        pcoef[perm] = np.abs(coef_).max()
        plop[perm] = coef_

    threshold = np.sort(pcoef)[(1 - pval) * n_perm]

    return np.abs(coef) > threshold


def bootstrap_permutation_testing(X, y, n_bootstraps=100, n_perm=10000,
                                  pval=.05):
    """ """
    n_samples, n_features = X.shape
    alpha_max = np.max(np.dot(y, X)) / n_samples
    alpha = .1 * alpha_max
    clf = BootstrapLasso(alpha=alpha, n_bootstraps=n_bootstraps)
    coef = clf.fit(X, y).coef_
    pcoef = np.zeros(n_perm)
    for perm in range(n_perm):
        y_ = y * (2 * (np.random.randint(0, 2, n_samples) - .5))
        # alpha_ = .1 * np.max(np.dot(y_, X)) / n_samples
        # clf.alpha = alpha_
        coef_ = clf.fit(X, y_).coef_
        pcoef[perm] = np.abs(coef_).max()

    threshold = np.sort(pcoef)[(1 - pval) * n_perm]
    return np.abs(coef) > threshold


# data generation
n_bootstraps = 100
n_true = 5
n_features = 50
n_voxels = 200
n_perm = 400
n_samples = 50
n_targets = 1

np.random.seed([1])
X = np.random.randn(n_samples, n_features)
X[:, :n_true] = 0.5 * X[:, :n_true] + .5 * np.random.randn(n_samples, 1)
effects = .3 * np.vstack((np.ones((n_true, n_voxels)),
                          np.zeros((n_features - n_true, n_voxels))))
support = np.abs(effects) > 0
noise = np.random.randn(n_samples, n_voxels)
Y = np.dot(X, effects) + noise


def compute_scores(X, y):
    score_smr = accuracy_(support.T[-1], SMR(X, y))
    score_ss = accuracy_(support.T[-1], stability_selection(X, y))
    score_pt = accuracy_(support.T[-1], permutation_testing(
                X, y, n_perm=n_perm, pval=.05))
    #score_bpt = accuracy_(support.T[-1], bootstrap_permutation_testing(
    #            X, y, n_perm=n_perm, pval=.05)
    score_bpt = (0, 0)
    return np.concatenate((score_smr, score_ss, score_pt, score_bpt))

"""
scores = Parallel(n_jobs=1)(
    delayed(compute_scores)(X, y) for y in Y.T)

scores = np.array(scores)
#  np.savez('scores.npz', scores=scores)

plt.figure(figsize=(5, 4))
plt.boxplot(scores)
plt.show()
"""


def precision_recall_samples(X, y):
    pr_lasso = precision_recall(support.T[-1], lasso_coefs(X, y))
    stability = stability_selection(X, y, pi=None)
    estimated = []
    for st in np.unique(stability):
        estimated.append(stability > st - 1.e-12)
    pr_ss = precision_recall(support.T[-1], estimated)

    n_samples, n_features = X.shape
    alpha_max = np.max(np.dot(y, X)) / n_samples
    alpha = .1 * alpha_max
    clf = Lasso(alpha=alpha)
    abs_coef = np.abs(clf.fit(X, y).coef_)
    estimated = []
    for th in np.unique(abs_coef):
        estimated.append(abs_coef > th - 1.e-12)

    pr_pt = precision_recall(support.T[-1], estimated)
    clf = BootstrapLasso(alpha=alpha, n_bootstraps=n_bootstraps)
    abs_coef = np.abs(clf.fit(X, y).coef_)
    estimated = []
    for th in np.unique(abs_coef):
        estimated.append(abs_coef > th - 1.e-12)

    pr_bpt = precision_recall(support.T[-1], estimated)
    return pr_lasso, pr_ss, pr_pt, pr_bpt

# pr_lasso, pr_ss, pr_pt, pr_bpt = precision_recall_samples(X, Y.T[0])
prs = Parallel(n_jobs=1)(
    delayed(precision_recall_samples)(X, y) for y in Y.T)

pr_lasso = (np.concatenate([pr[0][0] for pr in prs]),
         np.concatenate([pr[0][1] for pr in prs]))
pr_ss = (np.concatenate([pr[1][0] for pr in prs]),
         np.concatenate([pr[1][1] for pr in prs]))
pr_pt = (np.concatenate([pr[2][0] for pr in prs]),
         np.concatenate([pr[2][1] for pr in prs]))
pr_bpt = (np.concatenate([pr[3][0] for pr in prs]),
         np.concatenate([pr[3][1] for pr in prs]))




from sklearn import neighbors
c_values = np.linspace(.1, 1., 41)
clf = neighbors.KNeighborsRegressor(200, weights='uniform')
lasso_curve = clf.fit(
    pr_lasso[0][:, np.newaxis], pr_lasso[1]).predict(c_values[:, np.newaxis])
ss_curve = clf.fit(
    pr_ss[0][:, np.newaxis], pr_ss[1]).predict(c_values[:, np.newaxis])
pt_curve = clf.fit(
    pr_pt[0][:, np.newaxis], pr_pt[1]).predict(c_values[:, np.newaxis])
bpt_curve = clf.fit(
    pr_bpt[0][:, np.newaxis], pr_bpt[1]).predict(c_values[:, np.newaxis])


plt.figure()
plt.plot(c_values, lasso_curve, linewidth=2, label='Lasso path')
plt.plot(c_values, ss_curve, linewidth=2, label='Stability selection')
plt.plot(c_values, pt_curve, linewidth=2, label='Lasso point estimate')
plt.plot(c_values, bpt_curve, linewidth=2, label='Bootstrapped Lasso')
plt.legend(loc=1)
plt.show()


