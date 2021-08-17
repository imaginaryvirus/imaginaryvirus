# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 23:15:09 2021

@author: ainer
"""
import numpy as np
import random


class LogisticRegression():

    def __init__(self, tol, alpha, max_iter=1000, random_state=None):
        self.tol = tol  # 梯度下降停止的阈值，与loss的值比较
        self.alpha = alpha  # 梯度下降的更新步长
        self.max_iter = max_iter
        self.random_state = random_state
        self.coef_ = None

    def sparse_y(self, y):
        temp = [[0, 1] if val == 1 else [1, 0] for val in y]
        temp = np.array(temp)
        # y->nx2的矩阵
        self.y_sparse = temp

    def cross_entropy(self,X, y):
        # 优化的目标是最小化负对数似然函数
        # 与最小化交叉熵等价
        n_sample, _ = X.shape
        exp_x = np.exp(np.dot(X, self.coef_))
        p_y0 = np.reciprocal(exp_x + 1)
        log_p_y0 = np.log(p_y0)
        p_y1 = 1 - p_y0
        log_p_y1 = np.log(p_y1)
        result = np.dot(self.y_sparse[:, 0].T, log_p_y0) +\
                 np.dot(self.y_sparse[:, 1].T, log_p_y1)
        return -result/n_sample

    def update_coef(self, grad):
        self.coef_ = self.coef_ - self.alpha*grad

    def compute_grad(self, x, y):
        # 注意这里x是样本xi, y是样本标签yi
        n_feature, _ = self.coef_.shape
        grad = np.zeros((1, n_feature))
        exp_x = np.exp(np.dot(x, self.coef_))
        p_y1 = 1 - 1/(exp_x + 1)
        temp = y - p_y1
        grad -= x*temp
        return grad.T

    def SGD_solver(self, X, y):
        # 随机梯度下降
        # 更新梯度时每次随机选取一个样本，优点计算速度快
        n_sample, n_feature = X.shape
        num = n_sample - 1
        self.sparse_y(y)
        pre_loss = self.cross_entropy(X, y)
        if self.random_state:
            random.seed(self.random_state)
        for _ in range(self.max_iter):
            idx = random.randint(0, num)
            # 用抽取的样本计算梯度
            grad = self.compute_grad(X[idx], y[idx])
            self.update_coef(grad)
            loss = self.cross_entropy(X, y)
            # 计算|pre_loss - loss|决定是否停止迭代
            if abs(pre_loss - loss) <= self.tol:
                break
            pre_loss = loss
            print("current loss:", loss)

    def predict(self, X):
        exp_x = np.exp(np.dot(X, self.coef_))
        p_y0 = np.reciprocal(exp_x + 1)
        p_y1 = 1 - p_y0
        yp = [0 if y0 >= y1 else 1 for y0, y1 in zip(p_y0, p_y1)]
        yp = np.array(yp).reshape(1, -1)
        return yp

    def fit(self, X, y):
        n_sample, n_feature = X.shape
        self.coef_ = np.zeros((n_feature, 1))
        self.SGD_solver(X, y)
        return self

