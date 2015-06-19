"""
This script replicates the work by Ng et al. IPMI 2015

We start with a simpler setting, based on Lasso.

Author: Bertrand Thirion, 2015 
"""

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, MultiTaskLassoCV
from sklearn import metrics
import matplotlib.pyplot as plt


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


def accuracy(support, coef):
    return accuracy_(support, np.abs(coef) > 0)


def stability_selection(X, y, n_bootstraps=100, pi=.75):
    n_samples, n_features = X.shape
    alpha_max = np.max(np.dot(y, X)) / n_samples
    alpha = .1 * alpha_max
    stability = np.zeros(n_features)
    for b in range(n_bootstraps):
        X_ = .5 * X * (1 + np.random.rand(n_features) > 0)
        mask = np.random.rand(n_samples) > 0
        X_, y_ = X_[mask], y[mask]
        coef = Lasso(alpha=alpha).fit(X_, y_).coef_
        stability += np.abs(coef) > 0
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
        coefs = []
        n_samples = len(y)
        for b in range(self.n_bootstraps):
            samples = np.random.randint(0, n_samples, n_samples)
            y_, X_ = y[samples], X[samples]
            coefs.append(clf.fit(X_, y_).coef_)

        coefs = np.array(coefs)
        self.coef_ = coefs.mean(0) / coefs.std(0)
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
n_features = 20
n_voxels = 10
n_perm = 400
n_samples = 100
n_targets = 1

np.random.seed([1])
X = np.random.randn(n_samples, n_features)
effects = 1. * np.vstack((np.ones((n_true, n_voxels)),
                          np.zeros((n_features - n_true, n_voxels))))
support = np.abs(effects) > 0
noise = np.random.randn(n_samples, n_voxels)
Y = np.dot(X, effects) + noise

score_smr, score_ss, score_pt, score_bpt = [], [], [], []
for y in Y.T:
    score_smr.append(accuracy_(support.T[-1], SMR(X, y)))
    score_ss.append(accuracy_(support.T[-1], stability_selection(X, y)))
    score_pt.append(accuracy_(support.T[-1], permutation_testing(
                X, y, n_perm=n_perm, pval=.05)))
    #score_bpt = np.zeros_like(score_pt)
    score_bpt.append(accuracy_(support.T[-1], bootstrap_permutation_testing(
                X, y, n_perm=n_perm, pval=.05)))

score_smr, score_ss, score_pt, score_bpt = (
    np.array(score_smr), np.array(score_ss), np.array(score_pt),
    np.array(score_bpt))

scores = np.hstack((score_smr, score_ss, score_pt, score_bpt))
np.savez('scores.npz', scores=scores)

plt.figure(figsize=(5, 4))
plt.boxplot(scores)
plt.show()
