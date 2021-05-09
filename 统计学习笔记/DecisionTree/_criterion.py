# -*- coding: utf-8 -*-
"""
Created on Sat May  8 21:17:13 2021

@author: alien
"""
from abc import ABC
from abc import abstractmethod
from collections import Counter
import numpy as np
import math

class BaseCriterion(ABC):
    
    @abstractmethod
    def imputy(self, Y, indexs):
        pass

class Gini(BaseCriterion):
    
    def imputy(self, Y, indexs):
        res = Counter(Y[indexs])
        pks = np.array(list(res.values()))/len(indexs)
        gini = 1-sum(pks**2)
        return gini
    
    def children_impurity(self, Y, indexs, N):
        # 返回gini index
        gini = self.imputy(Y, indexs)
        return gini/N*len(indexs)

class Entropy(BaseCriterion):

    def imputy(self, Y, indexs):
        res = Counter(Y[indexs])
        pks = np.array(list(res.values()))/len(indexs)
        pks = pks.reshape((1, len(pks)))
        pks = np.apply_along_axis(lambda x: x*math.log2(x), 0, pks)
        return -pks.sum()
    
    def children_impurity(self, Y, indexs, N):
        ent = self.imputy(Y, indexs)
        return ent/N*len(indexs)

class MSE(BaseCriterion):
    
    def imputy(self, Y, indexs):
        mean = Y[indexs].mean()
        return sum((Y[indexs]-mean)**2)
    
    def children_impurity(self, Y, indexs, N):
        return self.imputy(Y, indexs)