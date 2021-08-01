# MLE，MAP

贝叶斯概率公式：

<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps1.jpg" alt="img" style="zoom:67%;" /> 

<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps24.jpg" alt="img" style="zoom:67%;" />为似然项，<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps25.jpg" alt="img" style="zoom:67%;" />为后验概率，<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps26.jpg" alt="img" style="zoom:67%;" />为先验概率。

### MLE与MAP

MLE（极大似然估计），即最大化<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps24.jpg" alt="img" style="zoom:67%;" />，意义也很直观，若<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps27.jpg" alt="img" style="zoom:67%;" />则在数据为D的情况下，参数更可能是<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps28.jpg" alt="img" style="zoom:67%;" />。已知观察的数据为D时，找到参数最有可能的值<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps6.jpg" alt="img" style="zoom:67%;" />。

MLE：

<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps7.jpg" alt="img" style="zoom:67%;" /> 

MAP（最大后验估计），即最大化<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps8.jpg" alt="img" style="zoom:67%;" />。

MAP：

<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps9.jpg" alt="img" style="zoom:67%;" /> 

MAP等于MLE减去**先验概率**的对数值。

### MLE与交叉熵

交叉熵：
<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps10.jpg" alt="img" style="zoom:67%;" /> 

我们希望实际分布与预测分布尽可能接近，令<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps11.jpg" alt="img" style="zoom:67%;" />

<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps29.jpg" alt="img" style="zoom:67%;" /> 

​	<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps30.jpg" alt="img" style="zoom:67%;" /> 

N为样本量，<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps13.jpg" alt="img" style="zoom: 67%;" />。

所以MLE与交叉熵的优化目标一致。

### MLE与最小二乘估计

假设模型<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml23008\wps1.jpg" alt="img" style="zoom:67%;" />，并且<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml23008\wps2.jpg" alt="img" style="zoom:67%;" />，即<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml23008\wps3.jpg" alt="img" style="zoom:67%;" />。e为误差项。

最小二乘估计：

<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml23008\wps4.jpg" alt="img" style="zoom:67%;" /> 

 MLE:<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml23008\wps6.jpg" alt="img" style="zoom:67%;" /> 

​			 <img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml23008\wps7.jpg" alt="img" style="zoom:67%;" /> 

当模型符合条件时，MLE与最小二乘的优化目标一致。

### MAP与正则化

假设<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps14.jpg" alt="img" style="zoom:50%;" />服从标准正态分布。
<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps15.jpg" alt="img" style="zoom:67%;" />

<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps16.jpg" alt="img" style="zoom:67%;" /> 

<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps17.jpg" alt="img" style="zoom:67%;" /> 

与L2正则：<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps18.jpg" alt="img" style="zoom:67%;" />一致。

 

假设<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps19.jpg" alt="img" style="zoom:67%;" />服从拉普拉斯分布。
<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps20.jpg" alt="img" style="zoom:67%;" />

<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps21.jpg" alt="img" style="zoom:67%;" /> 

<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps22.jpg" alt="img" style="zoom:67%;" /> 

与L1正则：<img src="file:///C:\Users\ainer\AppData\Local\Temp\ksohtml17012\wps23.jpg" alt="img" style="zoom:67%;" />一致。

 

所以引入正则项相当于限制了参数分布，引入了先验知识。L2正则对应参数服从正态分布，L1正则对应参数服从拉普拉斯分布。

