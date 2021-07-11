《Effective Python》第二版

1. 用Python3开发项目。

2. 遵循PEP8风格指南。

3. 了解bytes与str的区别。

4. 用支持插值的f-string取代%以及str.format方法。

   %的写法复杂，容易遇到许多问题。str.format方法同样有和%一样的缺点，应该避免使用上述两种方式。f-string可以直接在格式说明里嵌入任意Python表达式。

5. 用辅助函数取代复杂表达式。

   对于复杂且需要重复使用的表达式，应该写到辅助函数中。

6. 把数据结构直接拆分到多个变量中，不要专门通过下标访问。

   拆分（unpacking），可以把数据结构的多个值分别赋给多个变量，这种写法更清晰且代码量通常较少。赋值操作的左边可以是任意深度的可迭代对象。

7. 尽量用enumerate取代range。

8. 使用zip函数同时遍历两个迭代器。

9. 不要在for与while循环后面写else块。

   在for/else，while/else结构中，若循环提前终止else块的代码不会被执行。对空白序列做for循环或者while的首次循环就是False，程序会运行else块。代码不易读，应避免这种写法。

10. 用赋值表达式减少重复代码。

   赋值表达式是Python3.8新引入的语法，用到海象操作符（walrus operator）:=给变量赋值，并且让这个值成为这条表达式的结果，于是可以利用这项操作来缩减代码。

   ```
   fresh_fruit = {
       'apple' = 0
   }
   # old situation
   count = fresh_fruit.get('lemon', 0)
   if count:
       make_lemonade(count)
   else:
       print("Out of stock.")
   # new situation
   if count := fresh_fruit.get('lemon', 0):
       make_lemonade(count)
   else:
       print("Out of stock.")
   ```

   也可以用于模拟switch/case，do/while结构的逻辑。

11. 对序列做切片。

    切片尽可能写的简单一些，从头开始选取就省略起始下标，选到序列末尾就省略终止下标。把切片防止赋值符号左侧可以将原列表这段范围内的元素用赋值符号右侧元素替换掉，但可能会改变长度。

12. 不要在切片里同时指定起始下标与步进。

    同时指定切片的起始值下标与步进理解起来会很困难。如果要指定步进值，那就省略起止下标，且最好采用正数作为步进值，尽量不用负数。如果必须同时使用三项指标，那就分两次做（一次隔位选取，一次做切割）。可以用itertools模块的islice方法。

13. 通过带星号的unpacking操作来捕获多个元素，不要用切片

    基本的unpacking操作（第6条）需要提前确定序列的长度。使用下标与切片让代码卡看起来混乱，且容易出错。

    `oldest, second_oldest, *others = [1, 2, 3, 4, 5]`

    `print(others)`输出: [3, 4, 5]。

    带星号的表达式可以出现在任意位置，但**必须与至少一个**普通的接收变量搭配。另外对于单层结构来说，同一级里最多只能出现一次*。如果要拆解的结构有很多层，同一级的不同部分里面可以各自出现带星号的unpacking操作。带星号的表达式总是形成一份**列表实例**，如果待拆分的序列中没有元素给它，那么列表就是空白的。

14. 用sort方法的key参数来表示复杂的排序逻辑。

    把辅助函数传递给key参数，sort根据函数返回值来排序。

    如果排序时要依赖的指标有很多项，可以把它们放在一个元组内，让key函数返回这样的**元组**，对于支持一元减操作的类型，可以单独给这项指标取反，让排序在这项指标按相反的方向处理。

    如果不支持一元减操作，可以多次调用sort方法，最次要的指标在第一轮处理，最重要的在最后处理。

15. 不要过分依赖给字典添加条目时所用的顺序。

    从Python3.7开始迭代标准字典时看到的顺序和键值对的插入顺序一致。

16. 用get处理key不在字典的情况，不要用in与KeyError。

    ```
    names = votes.get(key)
    if names is None:
    	votes[key] = names = []
        
    names.append(who)
    
    # 继续简化
    if (names := votes.get(key)) is None:
    	votes[key] = names = []
        
    names.append(who)
    ```

17. 用defauldict处理内部状态中的缺失元素，不要用setdefault。

    如果你管理的字典可能需要添加任意的键，那么应该考虑能否用内置的collections模块的defaultdict实例来解决问题。如果这种键名比较随意的字典是别人传给你的，无法把它创建成defaultdict，那么应该考虑用get方法访问其中的键值。

18. 利用`__missing__`构造依赖键的默认值

    例如要写一个程序，在文件系统内管理社交网络账号中的图片。这个程序应该用字典把路径与相关的句柄关联起来，方便读取并写入图像。

    ```
    def open_picture(profile_path):
    	try:
    		return open(profile_path, 'a+b')
    	except OSError:
    		print(f'Failed to open path {profile_path}')
    		raise
    
    
    class Pictures(dict):
    	def __missing__(self, key):
    		value = open_picture(key)
    		self[key] = value
    		return value
    ```

    如果创建默认值需要较大的开销，或者可能抛出异常，那就不适合用dict类型的setdefault方法实现。

    传给defaultdict的函数必须是**不需要参数的函数**，所以无法创建出需要依赖键名的默认值。

    如果要构造的**默认值必须根据键名来确定**，那么可以定义自己的dict子类并实现`__missing__`方法。

19. 不要把函数返回值拆分到三个以上的变量中。

    函数可以把多个返回值合起来通过一个元组返回，以便利用unpacking机制拆分。

    把返回的值拆分到四个或者四个以上的变量是很容易出错的，所以最好用一个轻便的类或者namedtuple实现。

20. 遇到意外状况时应该抛出异常，不要返回None。

    要用返回值None表示特殊情况是很容易出错的，因为这样的值在条件表达式里没办法与0和空白字符串之类的值区分开，这些值都相当与False。

    用异常表示特殊的情况，而不要返回None，让调用函数的程序根据文档里写的异常情况作出处理。

    通过类型注解可以明确禁止函数返回None，即使在特殊情况下，它也不能返回这个值。









