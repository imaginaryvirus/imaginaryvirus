# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:36:37 2021

@author: alien
"""
from collections import Counter
from _criterion import Gini, Entropy, MSE
from collections import defaultdict
import numpy as np
import copy


class Node():
    """
    决策树的节点。
    """
    def __init__(
                 self,
                 split_val=None,
                 feature=None,
                 samples=[],
                 left=None,
                 right=None,
                 val=None,
                 imputy=None
                 ):
        """
        Parameters
        ----------
        split_val : float, optional
            非叶子节点的最佳划分值. The default is None.
        feature : int, optional
            非叶子节点的划分特征索引. The default is None.
        samples : list/numpy array, optional
            X被划入该节点的样本索引数组. The default is [].
        left : Node, optional
            左叶子节点. The default is None.
        right : Node, optional
            右叶子节点. The default is None.
        val : float, optional
            节点的值，分类树取样本中最多的类，回归树取样本标签均值. The default is None.
        imputy : 纯度, optional
            节点样本的纯度. The default is None.

        Returns
        -------
        None.

        """
        self.split_val = split_val
        self.feature = feature
        self.samples = samples
        self.left = left
        self.right = right
        self.val = val
        self.leaf = None
        self.imputy = imputy

    def __str__(self):
        return "split feature:{0} split value:{1} node value:{2:.3f}".format(self.feature, self.split_val, self.val)


class DecisionTree():
 
    def __init__(self,
                 max_depth,
                 min_samples_split=1,
                 min_samples_leaf=1,
                 min_impurity_split=0,
                 max_leaf_nodes=None,
                 ccp_alpha=0.0,
                 criterion='gini',
                 objective='classifier'
                 ):
        """

        Parameters
        ----------
        max_depth : int
            决策树最大深度.
        min_samples_split : int, optional
            内部节点分裂所需的最少样本数. The default is 1.
        min_samples_leaf : int, optional
            分裂为叶子节点所需最小样本数. The default is 1.
        min_impurity_split : int, optional
            停止分裂的阈值，分裂的增益大于该值才分裂. The default is 0.
        max_leaf_nodes : int, optional
            叶子节点的最大数. The default is None.
        ccp_alpha : float, optional
            执行代价复杂度剪枝的阈值，min_alpha大于该值停止剪枝. The default is 0.0.
        criterion : str, optional
            样本纯度的计算方法，{'gini', 'entropy', 'mse'}. The default is 'gini'.
        objective : str, optional
            {'classifier', 'regressor'}
            决策树的类型. The default is 'classifier'.

        Returns
        -------
        None.

        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_split = min_impurity_split
        self.max_leaf_nodes = max_leaf_nodes if max_leaf_nodes else 2**max_depth
        self.ccp_alpha = ccp_alpha
        self.criterion = self.__criterion(criterion)  # 计算纯度的类
        self.objective = objective
        self.depth = 0
        self.leaf_num = 0
        self.root = None
        self.X = None
        self.Y = None

    def __criterion(self, criterion):
        """
        返回相应纯度计算标准的类。
        """
        if criterion == 'gini':
            return Gini()
        elif criterion == 'entropy':
            return Entropy()
        elif criterion == 'mse':
            return MSE()
        else:
            raise ValueError("Wrong criterion.")

    def __get_val(self, y):
        """
        返回节点的值。
        """
        if self.objective == 'classifier':
            counts = Counter(y).most_common(1)
            return counts[0][0]
        else:
            return y.mean()

    def __get_subsets(self, x, sample_indexs):
        """
        返回字典，key:xi，value:取xi的样本索引列表。

        """
        out_sets = defaultdict(list)
        for x_val, y in zip(x, sample_indexs):
            out_sets[x_val].append(y)
        return out_sets

    def __split(self, node, depth):
        """
        递归建立决策树，划分当前的node。

        Parameters
        ----------
        node : Node
            当前要划分的节点.
        depth : int
            当前节点的深度.

        Returns
        -------
        None.

        """
        imputy = self.criterion.imputy(self.Y, node.samples)
        node.imputy = imputy  # 存储该节点的imputy
        node.val = self.__get_val(self.Y[node.samples])  # 储存节点的值
        if depth < self.max_depth and self.leaf_num <= self.max_leaf_nodes-2:
            if len(node.samples) < self.min_samples_split or\
                    len(set(self.Y[node.samples])) == 1:
                # 节点的样本量是否高于最低划分量
                # 或者当前节点的样本已经是同一类了
                self.leaf_num += 1
                node.leaf = True
                return
            max_improve = 0
            max_imputy_ind = None
            split_val = 0
            best_left = []
            best_right = []
            N = len(node.samples)
            for i in range(self.X.shape[1]):
                # 遍历features寻找最优划分feature以及特征值
                values = list(set(self.X[node.samples, i]))
                if len(values) <= 1:
                    continue
                values.sort()  # 对feature的取值排序
                # 得到每个取值的索引列表
                subsets = self.__get_subsets(self.X[node.samples, i], node.samples)
                left_inds = []
                right_inds = []
                for j in range(len(values)):
                    right_inds += subsets[values[j]]
                for j in range(0, len(values)-1):
                    # 每次将一个更大值的样本集合并入左子节点，并从右子节点取出
                    left_inds += subsets[values[j]]
                    right_inds = right_inds[len(subsets[values[j]]):]
                    if len(left_inds) < self.min_samples_leaf or \
                            len(right_inds) < self.min_samples_leaf:
                            continue
                    left_imputy = self.criterion.children_impurity(self.Y, left_inds, N)
                    right_imputy = self.criterion.children_impurity(self.Y, right_inds, N)
                    # 计算不纯度的增益
                    imputy_improve = imputy - left_imputy - right_imputy
                    if imputy_improve > max_improve:
                        max_improve = imputy_improve
                        max_imputy_ind = i
                        best_left = left_inds.copy()
                        best_right = right_inds.copy()
                        split_val = (values[j]+values[j+1])/2
            if max_improve >= self.min_impurity_split and max_improve != 0:
                node.left = Node(samples=sorted(best_left))
                node.right = Node(samples=sorted(best_right))
                node.feature = max_imputy_ind
                node.split_val = split_val
                # 继续分裂
                self.__split(node.left, depth+1)
                self.__split(node.right, depth+1)
            else:
                # 停止分裂，该节点记为叶子节点
                self.leaf_num += 1
                node.leaf = True
        else:
            # 记为叶子节点
            self.leaf_num += 1
            node.leaf = True

    def fit(self, X, Y, eval_set=None):
        """
        根据数据X, Y 训练决策树。

        Parameters
        ----------
        X : numpy array
            特征矩阵.
        Y : numpy array
            标签数组.
        eval_set : list, optional
            [eval_X, eval_Y]
            当ccp_alpha>0，需要额外的验证集来剪枝. The default is None.
        Returns
        -------
        None.

        """
        self.X = X
        self.Y = Y
        self.root = Node(samples=list(range(len(X))))
        cur_node = self.root
        self.__split(cur_node, 1)
        if self.ccp_alpha > 0:
            if eval_set is None:
                raise ValueError("Missing eval set")
            # 执行代价复杂剪枝
            self.__pruning(eval_set)
        self.X = None
        self.Y = None
        return

    def predict(self, X):
        """
        返回X的预测。

        """
        if self.root is None:
            raise ValueError("Fit model before using predict")
        # 层级遍历
        queue = [self.root]
        # 初始化根节点样本集
        self.root.samples = np.array(list(range(len(X))))
        out_labels = np.zeros((X.shape[0], 1))
        while queue:
            cur_node = queue.pop(0)
            if cur_node.leaf is None:  # 非叶子节点
                feature = cur_node.feature
                split_val = cur_node.split_val
                # samples内小于划分值的索引
                left_tmp = np.where(X[cur_node.samples, feature] <= split_val)[0]
                # 在Y中的索引
                left_inds = cur_node.samples[left_tmp]
                right_tmp = np.where(X[cur_node.samples, feature] > split_val)[0]
                right_inds = cur_node.samples[right_tmp]
                if len(left_inds) == 0 or len(right_inds) == 0:
                    # X不能再划分时，直接分配为多数的类
                    out_labels[cur_node.samples] = cur_node.val
                    continue
                cur_node.left.samples = left_inds
                cur_node.right.samples = right_inds
                queue.append(cur_node.left)
                queue.append(cur_node.right)
            else:
                # 当前节点为叶子节点，到达该节点的样本记为相应的类型
                out_labels[cur_node.samples] = cur_node.val
        return out_labels

    def __count_RT(self, node):
        """
        计算非叶子节点node的叶子节点的ΣR(T)。
        """
        T = 0
        queue = [node]
        RT = 0
        while queue:
            cur_node = queue.pop(0)
            if cur_node.leaf is True:
                # 记录叶子节点imputy
                T += 1
                RT += cur_node.imputy
            else:
                queue.append(cur_node.left)
                queue.append(cur_node.right)
        return RT, T

    def __pruning(self, eval_set):
        """
        采用代价复杂度剪枝，需要给出验证集或者以cv的方式选择最优子树。
        当树对应的最小a值大于ccp_a时停止剪枝。
        sklearn以叶子节点总imputy替代误分类率，计算R(T)。
        a = R(t)-R(T)/(T-1)
        """
        y_p = self.predict(eval_set[0])
        y_t = eval_set[1]
        if y_p.shape != y_t.shape:
            y_t = y_t.reshape(y_p.shape)
        N = len(y_t)
        num_leaf = self.leaf_num
        if self.objective == 'classifier':
            best_score = 0
        else:
            best_score = -(sum((y_p-y_t)**2)/N)[0]
        best_tree = copy.deepcopy(self.root)  # 深拷贝
        min_alpha = 0
        while self.leaf_num > 1:
            # 当决策树只剩一个树桩停止剪枝
            # 层级遍历现在的树，算出每个非叶子节点的alpha值
            queue = [self.root]
            alphas = []
            nodes = []
            while queue:
                cur_node = queue.pop(0)
                if cur_node.leaf is None:
                    # 非叶子节点
                    Rt = cur_node.imputy
                    RT, T = self.__count_RT(cur_node)
                    alpha = (Rt-RT)/(T-1)
                    alphas.append(alpha)
                    nodes.append((cur_node, T))
                    queue.append(cur_node.left)
                    queue.append(cur_node.right)
            min_alpha = min(alphas)
            if min_alpha > self.ccp_alpha:
                # 或者min_alpha>ccp_alpha停止
                break
            node, T = nodes[alphas.index(min_alpha)]  # 本轮min_alpha对应的节点
            node.left = None
            node.right = None
            node.leaf = True
            node.split_val = None
            node.feature = None
            self.leaf_num -= (T-1)  # 更新叶子节点数
            y_p = self.predict(eval_set[0])  # 更新预测
            # 这里用了acc和mse，其他metric可以用sklearn计算
            if self.objective == 'classifier':
                score = sum([1 for yp, yt in zip(y_p, y_t) if yp == yt])/N
            else:
                score = -(sum((y_p-y_t)**2)/N)[0]
            if score > best_score:
                best_score = score
                best_tree = copy.deepcopy(self.root)
                num_leaf = self.leaf_num
        self.root = best_tree
        self.leaf_num = num_leaf

    def print_tree(self):
        """
        层级遍历打印决策树。

        """
        if self.root:
            queue = [self.root]
            depth = 1
            while queue:
                n = len(queue)
                for i in range(n):
                    cur_node = queue.pop(0)
                    print("depth:", depth, cur_node)
                    if cur_node.leaf is None:
                        queue.append(cur_node.left)
                        queue.append(cur_node.right)
                depth += 1
            self.depth = depth
