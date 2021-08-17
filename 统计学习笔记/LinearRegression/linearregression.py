# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 20:38:35 2021

@author: ainer
"""
import numpy as np

class LinearRegression():

    def __init__(self, penalty='l2', alpha=1.0, coef_init=None):
        # lenght = n_feature + 1
        # 偏置对应输入Xi0 = 1
        self.coef_ = None
        self.penalty = penalty
        self.alpha = alpha
        self.indexs = None
        self.__coef_length = None
        self.coef_init = coef_init

    def __init_coef(self, n_feature, X_dtype):
        if self.coef_init is None:
            # 默认是0
            self.coef_ = np.zeros((n_feature, 1), dtype=X_dtype, order='F')
        else:
            self.coef_ = self.coef_init.reshape((n_feature, 1))
        self.__coef_length = n_feature
            
    def fit(self, X, y):
        if X.ndim != 2:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_sample, n_feature = X.shape
        self.__init_coef(n_feature, X.dtype)
        if self.penalty == 'l2':
            self.ridge_solvor(X, y, n_feature)
        else:
            self.lasso_solver(X, y, n_feature)

    def predict(self, X):
        if X.shape[1] != self.__coef_length:
            raise ValueError("Input array shape dismatch." +
                             "Require (n, {}).".format(
                             self.__coef_length))
        yp = np.dot(X, self.coef_)
        return yp.reshape(-1, 1)

    def ridge_solvor(self, X, y, n_feature):
        n_sample, n_feature = X.shape
        A = np.dot(X.T, X) + self.alpha*np.eye(n_feature, n_feature)
        self.coef_ = np.dot(np.linalg.inv(A), np.dot(X.T, y))

    def lasso_solver(self, X, y, n_feature):
        n_sample, n_feature = X.shape
        if X.flags['F_CONTIGUOUS'] is False:
            # X存储方式改为列在内存中连续的方式
            X = np.asfortranarray(X)
        self.indexs = [True for i in range(n_feature)]
        for k in range(n_feature):
            self.coordinate_descent_task(X, y, k)

    def coordinate_descent_task(self, X, y, index):
        self.indexs[index] = False
        xw_out_k = np.dot(X[:, self.indexs], self.coef_[self.indexs])
        ak = 2*np.dot(X[:, index].T, (xw_out_k - y))
        bk = 2*np.dot(X[:, index].T, X[:, index])
        if ak > self.alpha:
            self.coef_[index] = -(ak - self.alpha)/bk
        elif ak < -self.alpha:
            self.coef_[index] = -(ak + self.alpha)/bk
        else:
            self.coef_[index] = 0
        self.indexs[index] = True

    def score(self, X, y):
        if X.ndim != 2:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_sample, n_feature = X.shape
        yp = self.predict(X)
        temp = yp - y
        return np.dot(temp.T, temp)/n_sample

