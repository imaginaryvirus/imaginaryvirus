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

21. 了解如何在闭包里面使用外围作用域中的变量。

    例如要对列表元素排序，且某些元素有更高的优先级。可以这样实现：

    ```
    def sort_priority(values, group):
    	def helper(x):
    		if x in group:
    			return (0, x)
    		return (1, x)
    	values.sort(key=helper)
    ```

    这个函数能实现此功能有如下原因：

    python支持闭包（closure），让定义在大函数内的小函数也能引用大函数之中的变量。

    函数在python内是头等对象（first-class object），所以可以直接引用它们、把它们赋值给变量，将它们当成参数传递给其他函数、在in表达式、if语句里面对它作比较，等等。闭包函数也是函数，所以可以传给sort函数作为参数。

    python在判断两个序列是否相等时，先比较0号位置元素，如果相等再比较1号位置元素，以此类推。所以helper函数返回一个元组，并把关键指标写为首个元素。

    在表达式中**引用某个变量**时，python解释器按照以下顺序，在各个作用域（scope）里面查找这个变量，以解析这次引用：

    1. 当前函数作用域。
    2. 外围作用域。
    3. 包含当前代码的那个模块对应的作用域。也叫全局域（global scope）。
    4. 内置作用域（built-in scope，也就是包含len,str等函数的作用域）。

    如果这些作用域内都没有定义名称相符的变量程序抛出NameError异常。

    对于**变量赋值**：

    如果变量已经定义在当前作用域，那么直接把新值赋值。如果当前作用域内没有这个变量，那么及时外围有同名变量，python还是把这次赋值操作当成变量的定义来处理，且把包含赋值操作的函数当成新定义变量的作用域，**不会影响到外围作用域的同名变量的值**。防止函数中的局部变量污染外围模块。

    使用nonlocal关键字可以把闭包内的数据赋值给闭包外的变量。除了特别简单的函数应该尽量少使用nonlocal语句。

22. 用数量可变的位置参数给函数设计清晰的参数列表。

    用def定义函数时，可以通过*args的写法让函数接受数量可变的位置参数。

    调用函数时，可以在序列左边加上操作符`*`，把其中的**元素当成位置参数**传给*args所表示的这部分。

    如果*操作符在生成器前，传递参数时，程序可能因为内存耗尽而崩溃。

    给接受*args参数的函数添加新的位置参数，可能导致含义排查的bug。

23. 用关键字参数来表示可选的行为。

    函数的参数可以由位置指定，也可以由关键字的形式指定。

    关键字可以让每个参数的作用更加明了，因为在调用函数时只按位置指定参数可能导致这些参数的含义不够明确。

    应该通过带默认值的关键字参数来扩展函数的行为。

    可选的关键字参数应该总是通过参数名来传递，而不是按位置传递。

24. 用None和docstring来描述默认值会变的参数。

    参数的默认值只会计算一次，也就是在系统把定义函数的模块加载进来的时候。所以如果默认值将来可能由调用方修改（例如[], {}）或者随着要调用时的情况变化（例如datatime.now()），那么程序会出现奇怪的效果。

    ```
    import json
    def decode(data, default={}):
    	try:
    		return json.loads(data)
    	except ValueError:
    		return default
    df1 = decode('bad data')
    df1['stuff'] = 5
    df2 = decode('also bad')
    print(df1)
    print(df2)
    print(df1 is df2)
    # True, df1和df2是同一份字典
    ```

    如果关键字参数的默认值属于这种会发生变化的值，那就应该写为None，并且在docstring内描述此时函数的默认行为。

    ```
    def decode(data, default={}):
    	try:
    		return json.loads(data)
    	except ValueError:
    		if default is None:
    			default = {}
    		return default
    ```

    默认值为None的关键字参数也可以添加类型注解。

25. 用只能以关键字指定和只能以位置传入参数来设计清晰的参数列表。

    Keyword-only argument是一种只能通过关键字指定而不能通过位置指定的参数。

    ```
    def self_division(number, divisor, *,
    				  ignore_overflow=False,
    				  ignore_zero_division=False):
    	pass
    ```

    参数列表的*符号把参数分为两组，左边是位置参数，**右边是只能用关键字指定的参数**。

    Position-only argument不允许通过关键字指定，要求必须按位置传递。在函数的参数列表中，位于**/符号的左侧**。(python3.8引入)

    在参数列表中，位于/和*之间的参数，可以按位置指定，也可以用关键词指定。

    ```
    def self_division(number, divisor, /,
    				  ndigits=10, *,
    				  ignore_overflow=False,
    				  ignore_zero_division=False):
    	pass
    ```

26. 用functools.wraps来定义函数修饰器。

    python中可以用修饰器（decorator）来封装某个函数，使得程序在执行这个函数之前与之后，分别运行某些代码。这意味着调用者传给函数的参数值，函数返回值，以及函数抛出的异常，都可以有修饰器访问并修改。

    修饰器可能会让那些利用introspection机制运行的工具（例如调试器，help()函数，对象序列化器）产生奇怪的行为。

    python内置的functools模块有个叫wraps的修饰器，可以帮助我们正确定义自己的修饰器，从而避开相关问题。

    ```
    from functools import wraps
    
    def trace(func):
    	@wraps(func)
    	def wrapper(*args, **kwargs):
    		result = func(*args, **kwargs)
    		print(f'{func.__name__}({args!r}, 					  {kwargs!r})->{result!r}')
    		return result
    	return wrapper
    ```

27. 用列表推导取代map与filter。

    列表推导要比内置的map，filter函数清晰，并且可以很容易地跳过原列表的某些数据，假如用map实现，那么必须搭配filter。

    字典与集合也可以通过推导来创建。

28. 控制推导逻辑的子表达式不要超过两个。

    推导的时候可以使用多层循环，每个循环可以带有多个条件。

    如果多个if条件出现在同一层循环内，那么它们之间默认是and关系。

    控制推导逻辑的子表达式不要超过两个，否则会很难读懂。

29. 用赋值表达式消除推导中的重复代码。

    ```python
    def get_batches(count, size):
    	return count//size
    
    found = {name: get_batches(stock.get(name, 0), 8) 
             for name in order
    		 if get_batches(stock.get(name, 0), 8)}
    ```

    get_batches()执行了两遍，实际上不需要把结果计算两次，并且如果这两个地方忘了同步更新，程序会出现bug。

    在推导过程中使用:=，可以解决这一问题。

    ```python
    found = {name: batches for name in order
    		  if (batches := get_batches(stock.get(name, 0), 8))}
    ```

    编写推导式与生成器表达式时，可以在描述条件的那一部分通过赋值表达式定义变量，并在其他部分复用该变量。

    不要在推导式与生成器表达式的非条件部分使用赋值表达使，会使:=左边的变量泄露到包含这条推导语句的作用域中。

30. 不要让函数直接返回列表，应该让它逐个生成列表内的值。

    如果函数要返回的是一个包含许多结果的列表，这种函数改用生成器来实现比较好。生成器由包含yield表达式的函数创建。

    ```python
    def index_words_iter(text):
        # 一个返回单词首字母索引的生成器
    	if text:
    		yield 0
    	for index, letter in enumerate(text):
    		if letter == ' ':
    			yield index + 1
    ```

    生成器函数所返回的迭代器可以产生一系列的值，每次产生的值都是由函数体的下一条的yield表达式所决定的。

    不管输入的数据量有多大，生成器函数每次都只需要根据其中的一小部分来计算当前的输出值。它**不用把整个输入值全部读取进来**，也不用一次就把所有的输出值都算好。

31. 谨慎地迭代函数所收到的参数。



