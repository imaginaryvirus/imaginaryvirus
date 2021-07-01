《Effective Python》第二版

1. 用Python3开发项目。

2. 遵循PEP8风格指南。

3. 用支持插值的f-string取代%以及str.format方法。

   %的写法复杂，容易遇到许多问题。str.format方法同样有和%一样的缺点，应该避免使用上述两种方式。f-string可以直接在格式说明里嵌入任意Python表达式。

4. 用辅助函数取代复杂表达式。

   对于复杂且需要重复使用的表达式，应该写到辅助函数中。

5. 把数据结构直接拆分到多个变量中，不要专门通过下标访问。

   拆分（unpacking），可以把数据结构的多个值分别赋给多个变量，这种写法更清晰且代码量通常较少。赋值操作的左边可以是任意深度的可迭代对象。

6. 尽量用enumerate取代range。

7. 使用zip函数同时遍历两个迭代器。

8. 不要在for与while循环后面写else块。

   在for/else，while/else结构中，若循环提前终止else块的代码不会被执行。对空白序列做for循环或者while的首次循环就是False，程序会运行else块。代码不易读，应避免这种写法。

9. 用赋值表达式减少重复代码。

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

10. 对序列做切片。

    切片尽可能写的简单一些，从头开始选取就省略起始下标，选到序列末尾就省略终止下标。把切片防止赋值符号左侧可以将原列表这段范围内的元素用赋值符号右侧元素替换掉，但可能会改变长度。

11. 不要在切片里同时指定起始下标与步进。

    同时指定切片的起始值下标与步进理解起来会很困难。如果要指定步进值，那就省略起止下标，且最好采用正数作为步进值，尽量不用负数。如果必须同时使用三项指标，那就分两次做（一次隔位选取，一次做切割）。可以用itertools模块的islice方法。

12. 









