首先明确两个概念：**子串与子序列**。
比如一个字符串“aaabbc”的一个子串为“aaa”。而“abc”是它的一个子序列。即子串必须是在字符串中连续的，而子序列可不连续，但在字符串中的索引需要是升序的（“cba”就不是子序列）。

- 回文问题

回文字符串指正序遍历和逆序遍历完全相同的字符串。
一个简单的题目认识回文字符串。

1. “回文串”是一个正读和反读都一样的字符串，比如“level”或者“noon”等等就是回文串。花花非常喜欢这种拥有对称美的回文串，生日的时候她得到两个礼物分别是字符串A和字符串B。现在她非常好奇有没有办法将字符串B插入字符串A使产生的字符串是一个回文串。你接受花花的请求，帮助她寻找有多少种插入办法可以使新串是一个回文串。如果字符串B插入的位置不同就考虑为不一样的办法。


   例如：
   A = “aba”，B = “b”。这里有4种把B插入A的办法：

   在A的第一个字母之前: "baba" 不是回文

   在第一个字母‘a’之后: "abba" 是回文

   在字母‘b’之后: "abba" 是回文

   在第二个字母'a'之后 "abab" 不是回文
   所以满足条件的答案为2

```
def solution(A, B):
    res = 0
    for i in range(len(A)+1):
        tmp = A[:i]+B+A[i:]
        if tmp == tmp[::-1]:  # 判断正序遍历和逆序是否相同
            res += 1
    return res
```

2. 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是回文串 。

   返回 s 所有可能的分割方案。


```python
def solution(s):
    def dfs(rest, path):
        # path存储回文子串
        if rest == "":
            res.append(path)
            return
        for i in range(len(rest)):
            # 返回 s 所有可能的分割方案，是一个排列问题
            if rest[:i+1] == rest[:i+1][::-1]:  # 只有当前子串是回文时，才计入path
                dfs(rest[i+1:], path + [rest[:i+1]])
            return
    res = []
    return dfs(s, [])
```


           返回 最小的分割次数。
s长度很长！这时再用暴力解法无法通过。
此时通过预处理(见check函数)以数组存储s[i][j]是否回文，避免反复判断是否回文。
状态：
f[i]表示以i结尾的子串分割为回文子串的最小分割次数。
转移方程：
if s[0, i]是回文字符串:
    f[i] = 0
else:
    此时要向前检查0到i-1的区间是否存在能与i形成回文子序列
    f[i] = 1 + f[i-1]
    表示i自己被单独分割一次的情况
    for j in range(i):
        if s[j, i]是回文子串：
            此时i-j作为一个回文串分割一次，加上0-j-1的最小分割次数
            f[i] = min(f[i], 1+f[j-1])

def solution(s):
    dp = [[0]*len(s) for _ in range(len(s))]
    dp[0][0] = True
    def check(s):
        # dp[i][j]存储从i-j的子串是否是回文的
​        for r in range(len(s)):  # 右边的索引
​            for l in range(r+1):  # l要取到r
​                if r == l:
​                    dp[l][r] = True
​                    continue
​                if s[r] == s[l]:
​                    if (l+1 <= r-1 and dp[l+1][r-1] is True) or r-l == 1:
​                        dp[l][r] = True
​                    else:
​                        dp[l][r] = False
​                else:
​                    dp[l][r] = False
​        return
​    min_split = [float('inf')]*len(s)  # 表示以i结尾的子串最少要分割几次
​    min_split[0] = 0
​    for i in range(1, len(s)):
​        if dp[0][i] is True:  # 0-i是一个回文串
​            min_split[i] = 0
​        else:  # 非回文，分割子区间，取最小的分割值，前面的回文串最长时
​            tmp = float('inf')
​            tmp = min(tmp, 1+min_split[i-1])  # i单独分割一次
            # 或者与前面的某个位置形成回文，一起分割
​            for j in range(1, i):
​                if dp[j][i] is True:
​                    tmp = min(tmp, 1 + min_split[j-1])
​            min_split[i] = tmp




	* 
字符串编辑问题


	1. 
小摩手里有一个字符串A，小拜的手里有一个字符串B，B的长度大于等于A，


           所以小摩想把A串变得和B串一样长，这样小拜就愿意和小摩一起玩了。
           而且A的长度增加到和B串一样长的时候，对应的每一位相等的越多，
           小拜就越喜欢。比如"abc"和"abd"对应相等的位数为2，为前两位。
           小摩可以在A的开头或者结尾添加任意字符，使得长度和B一样。
           现在问小摩对A串添加完字符之后，不相等的位数最少有多少位？
这类对字符串a加上一个字符的操作，基本上可以转化为对字符串b删去一个字符来求解。
问题转化为b的开头或结尾删去一个字母，使得不相等的位数最小。

def solution(A, B):
    dp = {}  # 存储b对应最小不相等位数
    def dfs(a, b):
        if len(a) == len(b):  # b的长度与a相等判断不相等位数
            n = len(a)
            st = 0
            count = 0
            while st<n:
                if a[st] != b[st]:
                    count += 1
                st += 1
            return count
        if dp.get(b):  # 避免重复搜索
            return dp[b]
        ans = float('inf')
        ans = min(ans, dfs(a, b[1:]), dfs(a, b[:-1]))  # 分别搜索b删除首字母和尾字母的结果
        dp[b] = ans
        return ans
    return dfs(A, B)
	1. 
字符串有三种编辑操作:插入一个字符、删除一个字符或者替换一个字符。


           给定两个字符串，编写一个函数判定它们是否只需要一次(或者零次)编辑。
           示例 1:
           输入:
           first = "pale"
           second = "ple"
           输出: True
           来源：力扣（LeetCode）
           链接：https://leetcode-cn.com/problems/one-away-lcci
因为编辑次数最大为1，按字符串的长度分情况：
1. 长度之差大于1，无法一次编辑使二者相同，false
2. 长度相等，仅能做替换操作，记录相同位点的不同字符数，若大于1，false
3. 长度之差为1，对长的字符串删除一个字符后看能否和短字符串相同

def oneEditAway(self, first: str, second: str) -> bool:
    gap = abs(len(first)-len(second))
    if gap > 1:
        return False
    elif first == second:
        return True
    if len(first) < len(second):  # 使first是较长的字符串
        first, second = second, first
    n = len(first)
    m = len(second)
    count = 0
    if n-m == 0:
        for i in range(n):
            if first[i] != second[i]:
                count += 1
        return count <= 1
    else:
        for i in range(n):  # 删除first中的每一字母
            if first[:i]+first[i+1:] == second:
                return True
        return False
	1. 
给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。


           你可以对一个单词进行如下三种操作：
           插入一个字符
           删除一个字符
           替换一个字符
           来源：力扣（LeetCode）
           链接：https://leetcode-cn.com/problems/edit-distance

一次编辑的升级版，类比两个字符串匹配。（先看下一节再回来看这一题）
状态：f[i][j]存储w1[0, i]与w2[0, j]变成相同字符串的最小编辑次数。
转移方程：
if w1[i] == w2[j]:
    本次不需要编辑
    f[i][j] = f[i-1][j-1]
else:
    可以进行三种操作
    1.替换w1[i] or w2[j], 使w1[i]=w2[j]
    2.之前说过插入一个字符等价于删除一个字符，插入和删除操作合并
     删除w1[i], 使w1[0, i-1]与w2[0, j]变成相同子符串
     删除w2[j], 使w1[0, i]与w2[0, j-1]变成相同子符串
    f[i][j] = min(f[i-1][j-1], f[i-1][j], f[i][j-1])+1
初始化：
    对i=0或j=0（此时相应字符串为空格）的最小编辑次数可以直接给出。

def minDistance(self, word1: str, word2: str) -> int:
    word1 = " " + word1
    word2 = " " + word2
    n = len(word1)
    m = len(word2)
    dp = [[float('inf')]*m for _ in range(n)] # 使字符串相同的最小编辑次数
    for i in range(n):
        # word1[0, i]与" "的最小编辑次数为i
        dp[i][0] = i
    for j in range(m):
        dp[0][j] = j
    for i in range(1, n):
        for j in range(1, m):
            if word1[i] == word2[j]:
                # 不用编辑
                dp[i][j] = dp[i-1][j-1]
            else:
                # 替换操作:dp[i-1][j-1]+1
                # 删除操作：
                #    删除word1一个字符, dp[i-1][j]+1
                #    删除word2一个字符, dp[i][j-1]+1
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])+1
    return dp[-1][-1]
	* 
字符串匹配问题


给定一个待匹配字符串string，与要在string中查找的pattern。要求返回是否存在pattern或数量或string的索引。
字符串匹配问题可以用KMP算法求解，或者通过动态规划求解。
动态规划法：
参考自AC_OIer的回答。
链接：https://leetcode-cn.com/problems/distinct-subsequences/solution/xiang-jie-zi-fu-chuan-pi-pei-wen-ti-de-t-wdtk/
对于两个字符串匹配，一个非常通用的状态定义如下：
定义 f[i][j]为考虑 string中[0, i] 的字符（闭区间），pattern中[0, j] 的字符是否匹配（存储的值视问题而定）。
对string和pattern首部都加上一个空格或者可以认为是正则中的'.'，方便后续操作：
string ："aaaabb" -> " aaaabb"
pattern: "bc" -> " bc"
并且初始化f[i][0] = True，视问题初始化，比如计数问题就是初始化为1，存在问题就记为True。表示pattern头部的空格可以与string任意位置匹配。
转移方程：
1. 若要求pattern在string以子串的形式出现
   if string[i] == pattern[j]: 
    此时只看string[0，i-1]与pattern[0, j-1]是否匹配
    f[i][j] = f[i-1][j-1]
   else: 
    f[i][j] = False
2. 若要求pattern在string以子序列出现
        f[i][j] = f[i-1][j] or (False if string[i] != pattern[j] else f[i-1][j-1])
        f[i-1][j]：因为子序列可以不连续，即string[i]可以不与pattern[j]匹配
        三元表达式对应string[i]与pattern[j]匹配的情况
    看一道具体的题目：
	1. 
    输入：s = "rabbbit", t = "rabbit"


           输出：3
           解释：
           如下图所示, 有 3 种可以从 s 中得到 "rabbit" 的方案。
           (上箭头符号 ^ 表示选取的字母)
           r a b b b i t
           ^^^^    ^^
           r a b b b i t
           ^^   ^ ^^^
           r a b b b i t
           ^^^    ^^^
           来源：力扣（LeetCode）
           链接：https://leetcode-cn.com/problems/distinct-subsequences
一道计数的问题，dp[i][j]存储的是s[0, i]与t[0, j]的匹配方案数。
同样的在s，t的开头加上空格，初始化dp[i][0]=1。

def numDistinct(self, s: str, t: str) -> int:
    s = " "+s
    t = " "+t
    m = len(t)
    n = len(s)
    dp = [[0]*m for _ in range(n)]
    for i in range(n):
        dp[i][0] = 1
    # 因为t, s加了空格所以t[0]与s的任意位置i匹配都是1
    # dp[i][j] s[:i+1]与t[:j+1]的匹配数
    for i in range(1, n):  # 跳过空格
        for j in range(1, m):
            dp[i][j] = dp[i-1][j] + (0 if s[i] != t[j] else dp[i-1][j-1])
            # dp[i-1][j]不用s[i]的方案数
            # 0 if s[i] != t[j] else dp[i-1][j-1] 使用s[i]的方案数
    return dp[-1][-1]  # 返回s末尾与t末尾的匹配的方案数
	1. 
给定文本text和待匹配字符串pattern，二者皆只包含小写字母，并且不为空。


           在text中找出匹配pattern的最短字符串，匹配指按序包含pattern，但不要求pattern连续。
           如text为abaacxbcbbbbacc，pattern为cbc，text中满足条件的是abaacxbcbbbbacc下划线部分。
           输出最短匹配序列起止位置（位置下标从0开始），
           用空格分隔。若有多个答案，输出起止位置最小的答案；
           若无满足条件的答案，则起止均为-1。
拆解问题：首先完成子序列的匹配问题，然后在匹配的方案中找到距离最小的方案输出下标。

def solution(txt, pt):
    dp = {}  # 存储txt[0, i]是否匹配pt[0, j]
    n = len(txt)+1
    m = len(pt)+1
    txt = " "+txt
    pt = " "+pt
    for i in range(n):
        dp[i, 0] = True
    for i in range(1, n):
        for j in range(1, m):
            if txt[i] == pt[j]:
                dp[i, j] = dp.get((i-1, j), False) or dp.get((i-1, j-1), False)
            else:
                dp[i, j] = dp.get((i-1, j), False)
    pt_end = m-1
    min_gap = float('inf')
    res = []
    # 只看j=m-1的是否匹配，在匹配的结尾向前找起点
    for i in range(1, n):  # 不含" "
        if dp[i, pt_end]:  # s[0, i]能匹配到pt结尾
            cur_txt = i  # 现在txt匹配位置的结尾
            cur_pt = pt_end
            while cur_pt > 0: # 向前找txt起点
                if txt[cur_txt] == pt[cur_pt]:
                    cur_txt -= 1
                    cur_pt -= 1
                else:
                    cur_txt -= 1
            if i - cur_txt < min_gap:  # 比较gap的大小
                min_gap = i - cur_txt
                res = [cur_txt, i-1]
    return res if res != [] else [-1, -1]
	1. 
给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。


           '.' 匹配任意单个字符
           '*' 匹配零个或多个前面的那一个元素
           所谓匹配，是要涵盖 整个字符串 s的，而不是部分字符串。
           来源：力扣（LeetCode）
           链接：https://leetcode-cn.com/problems/regular-expression-matching
这里我的情况分的比较细，有更简洁的写法。
并且本题的dp数组初始化要考虑连续的c*的情况。

def isMatch(self, s: str, p: str) -> bool:
    # 分割字符串变成数组，并且把'*'和它对应的字符捆绑
    def split_string(s):
        out = []
        for i, c in enumerate(s):
            if c == '*':
                if out != []:
                    out[-1] = out[-1]+c  # 把*和相应字符捆绑
                else:
                    out.append(c)
            else:
                out.append(c)
        return out
    s = split_string(s)
    p = split_string(p)
    s = ['.'] + s
    p = ['.'] + p
    n = len(s)
    m = len(p)
    dp = [[False]*m for _ in range(n)]
    # dp[i][j]表示s[:i+1]与p[:j+1]是否匹配
    dp[0][0] = True
    for j in range(1, m):
        if '*' not in p[j]:  # 如aaaa匹配c*b*a*时，即出现连续的x*需要将对应位置赋值为True
            continue
        else:
            dp[0][j] = dp[0][j-1]  # 只有从1开始连续的x*才被赋值True
    for i in range(1, n):
        for j in range(1, m):
            if "*" not in p[j] and '.' not in p[j]:  # 最普通的匹配
                if s[i] == p[j]:
                    dp[i][j] =  dp[i-1][j-1]  # 看上一次位置的匹配情况
                else:
                    dp[i][j] = False
            elif "*" in p[j] and '.' not in p[j]:  # 普通字母+*
                c = p[j][0]  # 字母
                tmp = False
                if s[i] == c:
                    tmp = dp[i-1][j]  # 匹配一个末尾字符
                tmp = tmp or dp[i][j-1]  # 不使用*
                dp[i][j] = tmp
            elif '.' == p[j]:  # 只有.
                dp[i][j] = dp[i-1][j-1]
            else:  # p[j] = .*
                dp[i][j] = dp[i-1][j] or dp[i][j-1]
        return dp[-1][-1]
KMP算法
参考：
作者：labuladong https://zhuanlan.zhihu.com/p/83334559
python实现

import numpy as np
class KMP():
    # 思想：1.根据string[i]与pattern[j]的匹配情况退回到pattern的某个前缀
    #      从而避免每次都从头匹配pattern。
    #      2.使得string的指针不退回。
    #      3.dp数组只与pattern有关，与string无关。
    def __init__(self, pt):
        self.dp = np.zeros((len(pt), 256))
        # dp[j, c]表示pt状态为j遇到string的字符为c时应转移的状态
        self.pt = pt
    
    def kmp(self):
        # x:用于寻找应该退回的状态，叫影子状态
        #   总是落后状态j一个状态，与j具有最长的相同前缀。
        pt = self.pt
        m = len(self.pt)
        dp[0][ord(pt[0])] = 1  # 初始化匹配的情况
        x = 0
        for j in range(1, m):
            for c in range(256): # c表示ASCII码
                if c == ord(pt[j]):
                    # 当二者匹配时，状态向后转移
                    dp[j][c] = j+1
                else:
                    # 此时pt要退回到之前的某个状态
                    dp[j][c] = dp[x][c]
                    # kmp要使pt指针的回退尽可能小，所以通过x查询相同的前缀时
                    # x遇到c的转移状态。
            # 更新x
            x = dp[x][ord(pt[j])]
    
    def search(self, string):
        m = len(self.pt)
        pt_st = 0
        for i, c in enumerate(string):
            pt_st = self.dp[pt_st][ord(c)]  # 通过dp判断pt的下一个状态
            if pt_st == m:
                return i-m+1  # 可以匹配到pt[m-1], 返回下标
        return None