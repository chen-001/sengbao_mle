# sengbao_mle 
#### **翻新的最大似然估计框架** 
#### **copied and renewed maximum likelihood estimation package from [python-mle](https://github.com/ibab/python-mle)**
***

### 搬运与翻新说明
* 本仓库搬运自[ibab/python-mle](https://github.com/ibab/python-mle)

* 由于旧仓库最后一次更新于2016年2月19日，其中一些结构安排和依赖库已经发生了变化，因此笔者将其翻新后，重新发布到pypi上，便于直接安装调用

### 全新大版本📢
* v0.0.1 — 2022.10.24
> 搬运并翻新啦，version0恢复了原仓库的功能～将来会不断完善并添加新功能的🥳


### 安装&使用指南🎯
1. 安装
> 使用`pip install sengbao_mle`命令进行安装
2. 使用
>* **导入框架** 
>```python
>import sengbao_mle as m
>```
>* **使用实例1——原库示例（翻新后）**
>```python
>import numpy as np
>import sengbao_mle as m
>
># Define model
>x = m.var('x', observed=True, vector=True)
>y = m.var('y', observed=True, vector=True)
>
>a = m.var('a')
>b = m.var('b')
>sigma = m.var('sigma')
>
>model = m.Normal(y, a * x + b, sigma)
>
># Generate data
>xs = np.linspace(0, 2, 20)
>ys = 0.5 * xs + 0.3 + np.random.normal(0, 0.1, 20)
>
># Fit model to data
>result = model.fit({'x': xs, 'y': ys}, {'a': 1, 'b': 1, 'sigma': 1})
>print(result)
>```
>* **使用实例2——自定义分布**
>```python
>import numpy as np
>import sengbao_mle as m
>import theano.tensor as T
>
># 以Subbotin分布为例
>class Subbotin(m.Model):
>   def __init__(self,x,af,*args,**kwargs):
>       # 自定义的分布类型，均需继承自m.Model
>       super(Subbotin, self).__init__(*args,**kwargs)
>       # 运算都需要采用theano.tensor中的运算函数
>       mm=T.mean(x)
>       xg=T.mean(T.abs_(x-mm)**af)**(1/af)
>       # 写入该分布的对数概率密度函数
>       self._logp=m.distributions.bound(T.log(
>           T.exp(-((T.abs_((x-mm)/xg))**af)/af)/(2*xg*(af**(1/af))*T.gamma(1+1/af))
>       ))
>       # 添加样本点
>       self._add_expr('x',x)
>       # 添加要拟合的参数
>       self._add_expr('af',af)
>
># 样本点
>x = m.var('x', observed=True, vector=True)
># 目标参数
>af=m.var('af')
>model=Subbotin(x,af)
>
>xs=np.random.normal(-3,3,(1000,))
># 给目标参数设置初始值
>result = model.fit({'x': xs}, {'af': 1})
>print(result)
>```
>
>* **使用实例3——遍历初始值**
>```python
># 由于目标参数的初始值的设定，对能否拟合成功有着较大影响，因此在拟合失败时，可以考虑便利初始值，以寻找能拟合成功的初始参数
>import numpy as np
>import sengbao_mle as m
>import theano.tensor as T
>
># 以Subbotin分布为例
>class Subbotin(m.Model):
>   def __init__(self,x,af,*args,**kwargs):
>    	 # 自定义的分布类型，均需继承自m.Model
>       super(Subbotin, self).__init__(*args,**kwargs)
>       # 运算都需要采用theano.tensor中的运算函数
>       mm=T.mean(x)
>       xg=T.mean(T.abs_(x-mm)**af)**(1/af)
>       # 写入该分布的对数概率密度函数
>       self._logp=m.distributions.bound(T.log(
>       T.exp(-((T.abs_((x-mm)/xg))**af)/af)/(2*xg*(af**(1/af))*T.gamma(1+1/af))
>       ))
>       # 添加样本点
>       self._add_expr('x',x)
>       # 添加要拟合的参数
>       self._add_expr('af',af)
>
># 样本点
>x = m.var('x', observed=True, vector=True)
># 目标参数
>af=m.var('af')
>model=Subbotin(x,af)
>
>xs=np.random.normal(-3,3,(1000,))
># 遍历能拟合成功的初始参数值
>rs=[]
>for af in range(-100,100):
>		result = model.fit({'x': xs}, {'af': af})
> 		if result.success is True:
>				rs.append({af:result})
>af=np.mean([i.values().x['af'] for i in rs])      	
>print(af)
>```

#### 相关链接🔗
* [pypi](https://pypi.org/project/sengbao-mle/)