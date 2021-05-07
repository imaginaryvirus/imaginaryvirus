数位DP是求解范围区间[0，A]（或[A, B]， 此时求出B的解减去A的解即可）内符合条件的数字的个数。

条件一般与数字的组成有关。

[剑指 Offer 43. 1～n 整数中 1 出现的次数](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)

输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。

例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。

比如n取23456, 暴力解法一个个判断1~23456的数字显然效率很低。

通常采用记忆化搜索（dfs）求解。

通过遍历23456的每一位枚举每位可能的取值中符合条件的个数，配合记录每一位的**状态**以及相应的取值，

当枚举到记录过的状态时，便可以直接取相应的值。

这里给出解法。

```python
def solution(n):
    s = [int(c) for c in str(n)]
    length = len(s)
    dp = {}
    def dfs(pos, pre, lead, lim):
        # pos: 当前的位数
        # pre: 记录高位出现1的次数
        # lead: 前面的高位是否都是0，如000[pos]xxxx
        # lim: 当前位的取值是否是最大值
        if pos>=length:
            return pre
        if lead==0 and lim==0 and dp.get((pos, pre)):
            return dp[pos, pre]
        top = s[pos] if lim == 1 else 9
        ans = 0
        for i in range(top+1):
            if i == 0 and lead == 1:
                ans += dfs(pos+1, 0, lead, i == top and lim)
            elif i != 0 and lead == 1:
                if i == 1:
                    ans += dfs(pos+1, 1, 0, i == top and lim)
                else:
                    ans += dfs(pos+1, 0, 0, i == top and lim)
            else:
                if i == 1:  # 出现1，pre加1 
                    ans += dfs(pos+1, pre+1, lead, i==top and lim)
                else:
                    ans += dfs(pos+1, pre, lead, i==top and lim)
        if lim == 0 and lead == 0:
            dp[pos, pre] = ans
        return ans
    return dfs(0, 0, 1, 1)
```

配合题解来理解，首先是记忆化搜索为什么能够得到正确的答案？

比如现在搜索到了002xx，如果不采用记忆化搜索而是直接dfs枚举十位个位所有可能的取值有100种可能。

假设我们通过枚举得到了002xx中1出现的次数是m，那么003xx, 004xx, ..., 009xx，1出现的次数，就不必再通过枚举求解了，因为[2-9]**不含有1**，00[2-9]xx是否符合条件只与**xx**位有关。

所以从高位[0-2]xxxx开始枚举每位可能的取值，遇到以前记录到**相同的情况**直接计数即可。

所以需要记录取值产生的位数的**具体情况(pos和pre)**。并且当前位的取值是最大值（lim=1）或者当前位的高位是前导零（lead=1）时不能取用以前的记录值。

如当前枚举到234xx（pos=2, pre=0），此时4是百位能取的最大值，十位能取的值的范围是[0-5]，显然比之前记录的233xx（pos=2, pre=0）的取值范围小，含有1的数字更少。所以lim=1时，只能继续枚举后续位的值。

下面给出更通用的模板，参考了[Mathison](https://www.luogu.com.cn/blog/virus2017/shuweidp)的笔记。

```python
def dfs(pos, pre, st, lead, limit):
	# st: 记录状态，根据题目需要记录的状态可能会更多，多加几个记录状态的变量即可
    if pos >= length:
        return 返回值视题目而定
    if dp[pos][pre][st] != -1 and (limit != 1) and (lead !=1 ):
        # dp[pos][pre][st]被记录过了，且前一位不是最高位，前一位不是前导零
        return dp[pos][pre][st]  # 返回之前记录的值
    count = 0  # 暂时记录当前的方案数
    top = a[pos] if limit == 1 else 9  # top：当前位能取到的最大值
    for i in range(top+1):
    	# 每种情况pre, st的取值根据题目确定，这里以？代替
        # 有前导0并且当前位也是前导0
        if i == 0 and lead == 1:
            count += dfs(pos+1, ?, ?, lead, i==res and limit) # limit=0
        # 有前导0但当前位不是前导0，
        elif i != 0 and lead == 1:
            count += dfs(pos+1, ?, ?, 0, i==res and limit)
        elif(根据题意而定的判断):
            count += dfs(pos+1, ?, ?, lead, i==res and limit)
    if limit != 0 and lead != 0:  # 前一位不是最高位且前一位不是前导零
        dp[pos][pre][st] = count  # 记录当前状态方案数记录
    return count
```

[357. 计算各个位数不同的数字个数](https://leetcode-cn.com/problems/count-numbers-with-unique-digits/)

给定一个**非负**整数 n，计算各位数字都不同的数字 x 的个数，其中 0 ≤ x < 10^n。

套模板，用pre记录前面出现过的数字。这里pre用异或运算，来检查数字是否重复。

0-9的数字，用10位的二进制表示，1<<i, i∈[0-9]。

假设前面出现过1,3两数，pre: 000 000 101 0

此时又出现了3，pre^1<<3 = 000 000 001 0 < pre 

*注意这里给出的n是10的幂

```python
def solution(n):
	if n == 0:
        return 1
    dp = {}
    s = [9]*n
    res = []
    def dfs(pos, pre, lead, lim):
        # 用2进制记录出现数字出现的次数，000 000 000 0共10位, 
        # 0->01 1->10, 2->100
        # pre|(1<<i)
        # pre^i < pre 说明i出现过，剪枝，0要特殊处理
        if pos >= n:
            return 1
        if pre != 0 and dp.get((pos, pre)) and lim == 0 and lead == 0:
            return dp[pos, pre]
        top = s[pos] if lim == 1 else 9
        ans = 0
        for i in range(top+1): # 能取到的最高位数
            if pre^(1<<i) < pre: # 剪枝
                continue
            if i == 0 and lead == 1: #前导零与初始最高位
                ans += dfs(pos+1, pre, 1, lim and i==top)
            elif i != 0 and lead == 1:  # 前面是0，本位不为0
                ans += dfs(pos+1, pre^(1<<i), 0, lim and i==top)
            else: # 前一位不是0
                ans += dfs(pos+1, pre^(1<<i), lead, lim and i==top)
        if lim != 1 and lead == 0 and pre != 0:  # 前一位不是最高位且前一位不是前导零
            dp[(pos, pre)] = ans 
        return ans
    return dfs(0, 0, 1, 1)  # 初始时，lead, lim都是1
```

[1012. 至少有 1 位重复的数字](https://leetcode-cn.com/problems/numbers-with-repeated-digits/)

思路同上一题，以pre记录出现过的数字，st记录重复数字个数。

```python
def solution(N):
	if N<=10:
        return 0
    s = [int(num) for num in str(N)]
    length = len(s)
    dp = {}
    def dfs(pos, pre, st, lead, lim):
        if pos >= length:
            return 1 if st>0 else 0
        if lead == 0 and lim is False and dp.get((pos, pre, st)):
            return dp[pos, pre, st]
        top = s[pos] if lim is True else 9
        ans = 0
        for i in range(top+1):
            flag = False
            if pre^(1<<i) < pre:
                flag = True
            if i == 0 and lead == 1: # 前导零
                ans += dfs(pos+1, pre, 0, lead, i==top and lim)
            elif i != 0 and lead == 1: # 第一位非零
                ans += dfs(pos+1, pre^(1<<i), 0, 0, i==top and lim)
            else:
                if flag is True: # 与之前的记录数字有重复, pre不异或避免消去了本数字
                    ans += dfs(pos+1, pre, st+1, lead, i==top and lim)
                else:
                    ans += dfs(pos+1, pre^(1<<i), st, lead, i==top and lim)
        if lead == 0 and lim is False:
            dp[(pos, pre, st)] = ans
        return ans
    return dfs(0, 0, 0, 1, 1)
```

总结一下数位DP主要是模板，加上各种条件的判断，pre, st的取值要选好。



