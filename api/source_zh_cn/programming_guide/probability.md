# 深度概率编程库编程指南

<!-- TOC -->

- [深度概率编程库编程指南](#深度概率编程库编程指南)
    - [概率分布： mindspore.nn.probability.distribution](#概率分布-mindsporennprobabilitydistribution)
        - [概率分布类](#概率分布类)
            - [Distribution基类](#distribution基类)
            - [伯努利分布Bernoulli](#伯努利分布bernoulli)
            - [指数分布Exponential](#指数分布exponential)
            - [几何分布Geometric](#几何分布geometric)
            - [正态分布Normal](#正态分布normal)
            - [均匀分布Uniform](#均匀分布uniform)
        - [概率分布类在PyNative模式下的应用](#概率分布类在pynative模式下的应用)
        - [概率分布类在图模式下的应用](#概率分布类在图模式下的应用)
    - [概率分布映射： mindspore.nn.probability.bijector](#概率分布映射-mindsporennprobabilitybijector)
        - [Bijector类接口设计](#bijector类接口设计)
            - [Bijector基类](#bijector基类)
            - [幂函数变换映射PowerTransform](#幂函数变换映射powertransform)
            - [指数变换映射Exp](#指数变换映射exp)
            - [标量仿射变换映射ScalarAffine](#标量仿射变换映射scalaraffine)
            - [Softplus变换映射Softplus](#softplus变换映射softplus)
        - [PyNative模式下调用Bijector实例](#pynative模式下调用bijector实例)
        - [图模式下调用Bijector实例](#图模式下调用bijector实例)
        - [TransformedDistribution类接口设计](#transformeddistribution类接口设计)
        - [PyNative模式下调用TransformedDistribution实例](#pynative模式下调用transformeddistribution实例)
        - [图模式下调用TransformedDistribution实例](#图模式下调用transformeddistribution实例)
    - [深度概率网络：mindspore.nn.probability.dpn](#深度概率网络mindsporennprobabilitydpn)
        - [VAE](#vae)
        - [ConditionalVAE](#conditionalvae)
    - [概率推断算法：mindspore.nn.probability.infer](#概率推断算法mindsporennprobabilityinfer)
    - [贝叶斯层：mindspore.nn.probability.bnn_layers](#贝叶斯层mindsporennprobabilitybnn_layers)
        - [定义模型](#定义模型)
        - [训练模型](#训练模型)
    - [贝叶斯转换：mindspore.nn.probability.transform](#贝叶斯转换mindsporennprobabilitytransform)
        - [定义DNN模型](#定义dnn模型)
        - [定义损失函数和优化器](#定义损失函数和优化器)
        - [实例化TransformToBNN](#实例化transformtobnn)
            - [功能一：转换整个模型](#功能一转换整个模型)
            - [功能二：转换指定类型的层](#功能二转换指定类型的层)
    - [贝叶斯工具箱：mindspore.nn.probability.toolbox](#贝叶斯工具箱mindsporennprobabilitytoolbox)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/tree/master/api/source_zh_cn/programming_guide/probability.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概率分布： mindspore.nn.probability.distribution

概率分布是概率编程的基础。**Distribution** 类提供多样的概率统计接口, 例如概率密度函数 *pdf* 、累积密度函数*cdf* ，散度计算 *kl_loss* ，抽样 *sample* 等。现有的概率分布实例包括高斯分布，伯努利分布，指数型分布，几何分布和均匀分布。

### 概率分布类

- `Distribution`: 所有概率分布的基类。

- `Bernoulli`: 伯努利分布。参数为试验成功的概率。

- `Exponential`： 指数型分布。参数为率参数。

- `Geometric`: 几何分布。参数为一次伯努利试验成功的概率。

- `Normal`: 正态（高斯）分布。参数为均值和标准差。

- `Uniform`: 均匀分布。参数为数轴上的最小值和最大值。

#### Distribution基类

`Distribution` 继承`Cell`，是所有概率分布的基类。构造函数参数包括:

- `seed`: 随机抽样种子。
- `dtype`: 分布的数据类型。
- `name`: 分布的名称。
- `param`: 分布初始化时所用参数。

接口介绍：`Distribution`类支持的函数包括 `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, `log_survival`, `mean`, `sd`, `var`, `entropy`, `kl_loss`, `cross_entropy`,和 `sample`。分布不同，所需传入的参数也不同。只有在派生类中才能使用，由派生类的函数实现决定参数。

- `prob` : 概率密度函数（pdf）/ 概率质量函数（pmf）
- `log_prob` ：对数似然函数。
- `cdf` : 累积分布函数（cdf）。
- `log_cdf` : 对数累积分布函数（cdf）。
- `survival_function` : 生存函数。
- `log_survival` : 对数生存函数。
- `mean` : 均值。
- `sd` : 标准差。
- `var` : 方差。
- `entropy` : 熵。
- `kl_loss` : Kullback-Leibler散度。
- `cross_entropy` : 两个概率分布的交叉熵。
- `sample` : 概率分布的随机抽样。

#### 伯努利分布Bernoulli
伯努利分布，继承自 `Distribution` 类。构造函数参数如下：
- `probs`: 伯努利试验成功的概率。
- `seed`: 随机种子。
- `dtype`: 分布类型。
- `name`: 分布名称。

Property:
- `Bernoulli.probs`: 伯努利试验成功的概率。

`Distribution` 基类调用 `Bernoulli` 中私有接口以实现基类中的公有接口。`Bernoulli` 支持的公有接口为

- `mean`, `mode`, `var`: 可选择传入 *probs1* 指定试验成功的概率。
- `entropy`：可选择传入 *probs1* 指定试验成功的概率.
- `cross_entropy`, `kl_loss`：必须传入 *dist* 和 *probs1_b* 。*dist*: 另一分布类型的名称，目前只支持此处为 *‘Bernoulli’* ， *probs1_b* 指定分布 *b* 的试验成功概率。可选择传入 *probs1_a* 分布 *a* 的参数。
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, `log_survival`: 必须传入 *value* 。 可选择传入 *probs* 指定试验成功的概率。
- `sample`: 可选择传入 *shape* 样本形状和 *probs1* 指定试验成功的概率。

#### 指数分布Exponential
指数分布，继承自`Distribution`类。其构造函数参数如下：
- `rate`: 分布的率参数。
- `seed`: 随机种子。
- `dtype`: 分布类型。
- `name`: 分布名称。

Property:
- `Exponential.rate`: 率参数。

`Distribution` 基类调用 `Exponential` 私有接口以实现基类中的公有接口。`Exponential` 支持的公有接口为

- `mean`, `mode`, `var`: 可选择传入 *rate* 指定率参数
- `entropy`：可选择传入 *rate* 指定率参数
- `cross_entropy`, `kl_loss`：。必须传入 *dist* 和 *rate_b*. *dist*: 另一分布类型的名称， 目前只支持此处为 *‘Exponential’* ，*rate_b* 指定分布 *b* 的率参数。可选择传入 *rate_a* 分布 *a* 的参数.
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, `log_survival`: 必须传入 *value*，可选择传入 *rate* 指定率参数。
- `sample`: 可选择传入 *shape* 样本形状，以及 *rate* 指定率参数。

#### 几何分布Geometric
几何分布，继承自`Distribution`类。其构造函数参数如下：
- `probs`: 伯努利试验成功的概率。
- `seed`: 随机种子。
- `dtype`: 分布类型。
- `name`: 分布名称。

Property:
- `Geometric.probs`: 伯努利试验成功的概率。

`Distribution` 基类调用 `Geometric` 中私有接口以实现基类中的公有接口。`Geometric` 支持的公有接口为

- `mean`, `mode`, `var`: 可选择传入 *probs1* 指定试验成功的概率。
- `entropy`：可选择传入 *probs1* 指定试验成功的概率.
- `cross_entropy`, `kl_loss`：必须传入 *dist* 和 *probs1_b* 。*dist*: 另一分布类型的名称，目前只支持此处为 *‘Geometric’* ， *probs1_b* 指定分布 *b* 的试验成功概率。可选择传入 *probs1_a* 分布 *a* 的参数。
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, `log_survival`: 必须传入 *value*。 可选择传入 *probs1* 指定试验成功的概率。
- `sample`: 可选择传入 *shape* 样本形状和 *probs1* 指定试验成功的概率。

#### 正态分布Normal
正态（高斯）分布，继承自 **Distribution** 类。其构造函数参数如下：
- `mean`: 分布均值。
- `sd`: 分布方差。
- `seed`: 随机种子。
- `dtype`: 分布类型。
- `name`: 分布名称。

**Distribution** 基类调用 **Normal** 中私有接口以实现基类中的公有接口。**Normal** 支持的公有接口为
- `mean`, `mode`, `var`: 可选择传入 *mean* 和 *sd* 指定分布的参数。
- `entropy`：可选择传入 *mean* 和 *sd* 指定分布的参数。
- `cross_entropy`, `kl_loss`：必须传入 *dist* ，*mean_b* 和 *sd_b*。*dist*: 另一分布类型的名称，目前只支持此处为 *‘Normal’* ，*mean_b* 和 *sd_b* 指定分布b的参数。可选择传入 *mean_a* 和 *sd_a* 指定分布的参数。
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, `log_survival`: 必须传入 *value*。可选择传入 *mean* 和 *sd* 指定分布的参数。
- `sample`: 可选择传入 *shape* 样本形状。可选择传入 *mean_a* 和 *sd_a* 指定分布的参数。

#### 均匀分布Uniform
均匀分布，继承自`Distribution`类。其构造函数参数如下：
- `low`: 最小值。
- `high`: 最大值。
- `seed`: 随机种子。
- `dtype`: 分布类型。
- `name`: 分布名称。

Property:
- `Uniform.low`: 最小值。
- `Uniform.high`: 最大值。

`Distribution` 基类调用 `Uniform` 以实现基类中的公有接口。`Uniform` 支持的公有接口为

- `mean`, `mode`, `var`: 可选择传入 *high* 和 *low* 指定分布的参数。
- `entropy`：可选择传入 *high* 和 *low* 指定分布的参数。
- `cross_entropy`, `kl_loss`：必须传入 *dist* ，*high_b* 和 *low_b* 。*dist* : 另一分布类型的名称，目前只支持此处为 *‘Uniform’* ，*high_b* 和 *low_b* 指定分布 *b* 的参数。可选择传入 *high_a* 和 *low_a* 指定分布的参数。
- `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, `log_survival`: 必须传入 *value* 。可选择传入 *high* 和 *low* 指定分布的参数。
- `sample`: 可选择传入 *shape* 。 可选择传入 *high* 和 *low* 指定分布的参数。

### 概率分布类在PyNative模式下的应用

`Distribution`子类可在 **PyNative** 模式下使用。
导入相关模块：

```python
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.context as context
import mindspore.nn.probability.distribution as msd
context.set_context(mode=context.PYNATIVE_MODE)
```
以 **Normal** 为例， 创建一个均值为0.0 标准差为1.0 的正态分布：
```python
my_normal = msd.Normal(0.0, 1.0, dtype=mstype.float32)
```
计算均值：
```python
mean = my_normal.mean()
print(mean)
```
输出为：
```
0.0
```
计算方差：
```python
var = my_normal.var()
print(var)
```
输出为：
```
1.0
```
计算熵：
```python
entropy = my_normal.entropy()
print(entropy)
```
```
1.4189385
```
计算 **pdf**:
```python
value = Tensor([-0.5, 0.0, 0.5], dtype=mstype.float32)
prob = my_normal.prob(value)
print(prob)
```
输出为：
```
[0.35206532, 0.3989423, 0.35206532]
```
计算 **cdf**:
```python
cdf = my_normal.cdf(value)
print(cdf)
```
输出为：
```
[0.30852754, 0.5, 0.69146246]
```
计算 **kl_loss**:
```python
mean_b = Tensor(1.0, dtype=mstype.float32)
sd_b = Tensor(2.0, dtype=mstype.float32)
kl = my_normal.kl_loss('Normal', mean_b, sd_b)
print(kl)
```
输出为：
```
0.44314718
```

### 概率分布类在图模式下的应用

在图模式下，**Distribution**子类可用在网络中。
导入相关模块：
```python
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.context as context
import mindspore.nn.probability.distribution as msd
context.set_context(mode=context.GRAPH_MODE)
```
创建网络：
```python
# 网络继承nn.Cell
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        # 创建 Normal 实例
        self.normal = msd.Normal(0.0, 1.0, dtype=mstype.float32)
    
    #定义计算 
    def construct(self, value, mean, sd):
        pdf = self.normal.prob(value)
        kl = self.normal.kl_loss("Normal", mean, sd)
        return pdf, kl
```
调用网络
```python
net = Net()
value = Tensor([-0.5, 0.0, 0.5], dtype=mstype.float32)
mean = Tensor(1.0, dtype=mstype.float32)
sd = Tensor(1.0, dtype=mstype.float32)
pdf, kl = net(value, mean, sd)
print("pdf: ", pdf)
print("kl: ", kl)
```
输出为：
```
pdf: [0.3520653, 0.39894226, 0.3520653]
kl: 0.5
```

## 概率分布映射： mindspore.nn.probability.bijector

Bijector是概率编程的基本组成部分。Bijector描述了一种随机变量的变换方法，可以通过一个已有的随机变量X和一个映射函数f生成一个新的随机变量Y = f(x)。
`Bijector`提供了映射相关的四种变换方法。它可以当做算子直接使用，也可以作用在某个随机变量`Distribution`类实例上生成新的随机变量的`Distribution`类实例。

### Bijector类接口设计

#### Bijector基类

`Bijector`类是所有Bejictor的基类，继承自`Cell`基类。

* 基类构造函数__init__

  输入参数：
  `is_constant_jacobian`：映射函数的导数是否常数，只在创建衍生类的时候使用，默认false。
  `is_injective`：射函数是否是单设，只在创建衍生类的时候使用，默认true。
  `name`：Bijector的名字，创建派生类后由派生类的具体函数名决定，只在创建衍生类的时候使用，可以接受输入为str。默认None。
  `dtype`：Bijector对应的函数的类型（自变量和因变量的数据类型），创建派生类后由由派生类的具体函数决定，只在创建衍生类的时候使用，默认为None。
  `param`：Bijector内部的参数，创建派生类后由由派生类的具体函数参数决定，只在创建衍生类的时候使用，默认为None。

* 类特征函数property
`name`：无参函数，返回`name`的值。
`is_dtype`：无参函数，返回`dtype`的值。
`parameter`：无参函数，返回`parameter`的值。
`is_constant_jacobian`：无参函数，返回`is_constant_jacobian`的值。
`is_injective`：无参函数，返回`is_injective`的值。

* 映射函数
`forward (args)`：正向映射，创建派生类后由派生类的`_forward`决定参数。
`inverse (args)`：反向映射，创建派生类后由派生类的`_inverse`决定参数。
`forward_log_jacobian (args)`：正向映射的(log)导数，创建派生类后由派生类的`_forward_log_jacobian`决定参数。
`inverse_log_jacobian (args)`：反向映射的(log)导数，创建派生类后由派生类的`_inverse_log_jacobian`决定参数。

* `Bijector` 作为函数调用：
输入是一个`Distribution`类：生成一个`TransformedDistribution`**（不可在图内调用）**
其他输入：内部使用，不对外开放。

#### 幂函数变换映射PowerTransform
`PowerTransform`做如下变量替换：Y = g(X) = (1 + X * c)^(1 / c)。

* `PowerTransform`构造函数`__init__`
输入参数：
`power`：`PowerTransform`双射类对象的指数，可以接受输入为scalar(int or float), 默认0。
`name`：`PowerTransform`双射类对象的名字，可以接受输入为str。默认为“PowerTransform”。
`param`：`PowerTransform`内部的参数，创建派生类后由由派生类(exp)的具体函数参数决定，只在创建衍生类的时候使用。默认为None。

* 类特征函数property
`power`：无参函数，返回power的值。

* 映射函数
`forward (args)`：正向映射，输入为tensor。
`inverse (args)`：反向映射，输入为tensor。
`forward_log_jacobian (args)`：正向映射的(log)导数，输入为tensor。
`inverse_log_jacobian (args)`：反向映射的(log)导数，输入为tensor。

#### 指数变换映射Exp
`Exp`做如下变量替换：Y = g(X)= exp(X)。

* `Exp`构造函数`__init__`
输入参数：
`name`：Exp双射类对象的名字，可以接受输入为str。默认为“Exp”。

* 映射函数
`forward (args)`：正向映射，输入为tensor。继承自父类`PowerTransform`双射。
`inverse (args)`：反向映射，输入为tensor。继承自父类`PowerTransform`双射。
`forward_log_jacobian (args)`：正向映射的(log)导数，输入为tensor。继承自父类`PowerTransform`双射。
`inverse_log_jacobian (args)`：反向映射的(log)导数，输入为tensor。继承自父类`PowerTransform`双射。

#### 标量仿射变换映射ScalarAffine
`ScalarAffine`做如下变量替换：Y = g(X) = a * X + b。

* `ScalarAffine`构造函数`__init__`
输入参数：
`scale`：ScalarAffine的斜率，可以接受输入为float, 默认1.0。
`shift`：ScalarAffine的截距，可以接受输入为float, 默认0.0。
`name`：ScalarAffine的名字，可以接受输入为str。默认“ScalarAffine”。

* 类特征函数property
`scale`：无参函数，返回scale的值。
`shift`：无参函数，返回shift的值。

* 映射函数
`forward (args)`：正向映射，输入为tensor。
`inverse (args)`：反向映射，输入为tensor。
`forward_log_jacobian (args)`：正向映射的(log)导数，输入为tensor。
`inverse_log_jacobian (args)`：反向映射的(log)导数，输入为tensor。

#### Softplus变换映射Softplus
`Softplus`做如下变量替换：Y = g(X) = \frac{\log(1 + e ^ {kX})}{k}。

* `Softplus`构造函数`__init__`
输入参数：
`sharpness`：Softplus的斜率，可以接受输入为float, 默认1.0。
`name`：Softplus的名字，可以接受输入为str。默认“Softplus”。
* 类特征函数property
`sharpness`：无参函数，返回sharpness的值。
* 映射函数
  `forward (args)`：正向映射，输入为tensor。
  `inverse (args)`：反向映射，输入为tensor。
  `forward_log_jacobian (args)`：正向映射的(log)导数，输入为tensor。
  `inverse_log_jacobian (args)`：反向映射的(log)导数，输入为tensor。

### PyNative模式下调用Bijector实例

PyNative模式类似于命令行调用，可以即时输入函数及其参数，完成函数调用。PyNative模式为异步执行，仅在访问函数结果时函数会被调用。

在执行之前，我们需要导入需要的库文件包。双射类最主要的库是`mindspore.nn.probability.bijector`，导入后我们使用`msb`作为库的缩写并进行调用。

导入相关模块：
```python
import numpy as np
import mindspore.nn as nn
import mindspore.nn.probability.bijector as msb
import mindspore.context as context
from mindspore import Tensor
from mindspore import dtype
context.set_context(mode=context.PYNATIVE_MODE)
```

通常我们并不直接构造基类`Bijector`，而是直接构造派生的子类，并调用子类对应的函数接口。下面我们以`PowerTransform`为例。

创建一个`powertransform`对象，初始化参数中power=2。

构造`PowerTransform`:
```python
powertransform = msb.PowerTransform(power=2)
powertransform
```

输出:
```python
PowerTransform<power = 2>
```

接下来可以使用映射函数进行运算。

调用`forward`方法，计算正向映射:
```python
x = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
tx = Tensor(x, dtype=dtype.float32)
forward = powertransform.forward(tx)
foward
```

输出:
```python
Tensor(shape=[4], dtype=Float32, [ 2.23606801e+00  2.64575124e+00  3.00000000e+00  3.31662488e+00])
```

输入`inverse`方法，计算反向映射:
```python
inverse = powertransform.inverse(tx)
inverse
```

输出:
```python
Tensor(shape=[4], dtype=Float32, [ 1.50000000e+00  4.00000048e+00  7.50000000e+00  1.20000010e+01])
```

输入`forward_log_jacobian`方法，计算正向映射导数:
```python
forward_log_jaco = powertransform.forward_log_jacobian(tx)
forward_log_jaco
```

输出:
```python
Tensor(shape=[4], dtype=Float32, [-8.04718971e-01 -9.72955048e-01 -1.09861231e+00 -1.19894767e+00])
```

输入`inverse_log_jacobian`方法，计算正向映射导数:
```python
inverse_log_jaco = powertransform.inverse_log_jacobian(tx)
inverse_log_jaco
```

输出:
```python
Tensor(shape=[4], dtype=Float32, [ 6.93147182e-01  1.09861231e+00  1.38629436e+00  1.60943794e+00])
```

### 图模式下调用Bijector实例

在图模式下，`Bijector`子类可用在网络中。

导入相关模块：
```python
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.context as context
import mindspore.nn.probability.Bijector as msb
context.set_context(mode=context.GRAPH_MODE)
```

创建网络：
```python
# 网络继承nn.Cell
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        # 创建PowerTransform实例
        self.powertransform = msb.PowerTransform(power=2)
    
    #定义计算 
    def construct(self, value):
        forward = self.s1.forward(value)
        inverse = self.s1.inverse(value)
        forward_log_jaco = self.s1.forward_log_jacobian(value)
        inverse_log_jaco = self.s1.inverse_log_jacobian(value)
		return forward, inverse, forward_log_jaco, inverse_log_jaco
```
调用网络
```python
net = Net()
x = np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32)
tx = Tensor(x, dtype=dtype.float32)
forward, inverse, forward_log_jaco, inverse_log_jaco = net(tx)
print("forward: ", forward)
print("inverse: ", inverse)
print("forward_log_jaco: ", forward_log_jaco)
print("inverse_log_jaco: ", inverse_log_jaco)
```
输出为：
```python
forward:  [2.236068  2.6457512 3.        3.3166249]
inverse:  [ 1.5        4.0000005  7.5       12.000001 ]
forward_log_jaco:  [-0.804719   -0.97295505 -1.0986123  -1.1989477 ]
inverse_log_jaco:  [0.6931472 1.0986123 1.3862944 1.609438 ]
```

### TransformedDistribution类接口设计

`TransformedDistribution`继承自`Distribution`，是可通过映射f(x)变化得到的数学分布的基类。

* 构造函数`__init__`

  输入参数：
  `bijector`：分布的变换方法。可以接受输入为bijector的派生类，作为映射变换函数f。
  `distribution`：原始分布。可以接受输入为distribution的派生类，作为映射需要变换的分布类*X*。
  `seed`: 随机种子。
  `dtype`: 分布类型。
  `name`：分布名称。可以接受输入为str。默认为“transformed_distribution”。

* 类特征函数property
`bijector`：无参函数，返回分布的变换方法。
`distribution`：无参函数，返回原始分布。
`is_linear_transformation`：无参函数，返回线性变换标志。

* 接口函数
`cdf`：累积分布函数（cdf）。
`log_cdf`：对数累积分布函数（cdf）。
`survival_function`：生存函数。与 cdf 互补。
`log_survival`：对数生存函数。
`prob`：概率密度函数（pdf）/ 概率质量函数（pmf）。
`log_prob`：对数似然函数。
`sample`：随机取样。必须传入shape，类型tuple。
`mean`：无参数。只有当`Bijector.is_constant_jacobian=true`时可调用。

### PyNative模式下调用TransformedDistribution实例

`TransformedDistribution` 子类可在 **PyNative** 模式下使用。
在执行之前，我们需要导入需要的库文件包。

导入相关模块：
```python
import numpy as np
import mindspore.nn as nn
import mindspore.nn.probability.bijector as msb
import mindspore.nn.probability.distribution as msd
import mindspore.context as context
from mindspore import Tensor
from mindspore import dtype
context.set_context(mode=context.PYNATIVE_MODE)
```

构造一个`TransformedDistribution`实例，使用Normal分布作为需要变换的分布类，使用Exp作为映射变换，可以生成`LogNormal`分布。
```python
normal = msd.Normal(0.0, 1.0, dtype=dtype.float32)
exp = msb.Exp()
LogNormal = msd.TransformedDistribution(exp, normal, dtype=dtype.float32, seed=0, name="LogNormal")
LogNormal
```

输出：
```python
TransformedDistribution<
  (_bijector): Exp<power = 0>
  (_distribution): Normal<mean = 0.0, standard deviation = 1.0>
  >
```

可以对`LogNormal`进行概率分布计算。例如：

计算 **cdf** ：
```python
x = np.array([2.0, 5.0, 10.0], dtype=np.float32)
tx = Tensor(x, dtype=dtype.float32)
cdf = LogNormal.cdf(tx)
cdf
```

输出：
```python
Tensor(shape=[3], dtype=Float32, [ 7.55891383e-01  9.46239710e-01  9.89348888e-01])
```

计算 **log_cdf** ：
```python
x = np.array([2.0, 5.0, 10.0], dtype=np.float32)
tx = Tensor(x, dtype=dtype.float32)
log_cdf = LogNormal.log_cdf(tx)
log_cdf
```

输出：
```python
Tensor(shape=[3], dtype=Float32, [-2.79857576e-01 -5.52593507e-02 -1.07082408e-02])
```

计算 **survival_function** ：
```python
x = np.array([2.0, 5.0, 10.0], dtype=np.float32)
tx = Tensor(x, dtype=dtype.float32)
survival_function = LogNormal.survival_function(tx)
survival_function
```

输出：
```python
Tensor(shape=[3], dtype=Float32, [ 2.44108617e-01  5.37602901e-02  1.06511116e-02])
```

计算 **log_survival** ：
```python
x = np.array([2.0, 5.0, 10.0], dtype=np.float32)
tx = Tensor(x, dtype=dtype.float32)
log_survival = LogNormal.log_survival(tx)
log_survival
```

输出：
```python
Tensor(shape=[3], dtype=Float32, [-1.41014194e+00 -2.92322016e+00 -4.54209089e+00])
```

计算 **prob** ：
```python
x = np.array([2.0, 5.0, 10.0], dtype=np.float32)
tx = Tensor(x, dtype=dtype.float32)
prob = LogNormal.prob(tx)
prob
```

输出：
```python
Tensor(shape=[3], dtype=Float32, [ 1.56874031e-01  2.18507163e-02  2.81590177e-03])
```

计算 **log_prob** ：
```python
x = np.array([2.0, 5.0, 10.0], dtype=np.float32)
tx = Tensor(x, dtype=dtype.float32)
log_prob = LogNormal.log_prob(tx)
log_prob
```

输出：
```python
Tensor(shape=[3], dtype=Float32, [-1.85231221e+00 -3.82352161e+00 -5.87247276e+00])
```

调用取样函数 **sample** ：
```python
shape = ((3, 2))
sample = LogNormal.sample(shape)
sample
```

输出：
```python
Tensor(shape=[3, 2], dtype=Float32,
[[ 7.64315844e-01  3.01435232e-01]
 [ 1.17166102e+00  2.60277224e+00]
 [ 7.02699006e-01  3.91564220e-01]])
```

### 图模式下调用TransformedDistribution实例

在图模式下，**TransformedDistribution**子类可用在网络中。

导入相关模块：
```python
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype
import mindspore.context as context
import mindspore.nn.probability.Bijector as msb
import mindspore.nn.probability.Distribution as msd
context.set_context(mode=self.GRAPH_MODE)
```

创建网络：
```python
# 网络继承nn.Cell
class Net(nn.Cell):
    def __init__(self, shape, dtype=dtype.float32, seed=0, name='transformed_distribution'):
        super(Net, self).__init__()
        # 创建TransformedDistribution实例
        self.exp = msb.Exp()
        self.normal = msd.Normal(0.0, 1.0, dtype=dtype)
        self.lognormal = msd.TransformedDistribution(self.exp, self.normal, dtype=dtype, seed=seed, name=name)
        self.shape = shape
    
    #定义计算cdf，以及随机取样
    def construct(self, value):
        cdf = self.lognormal.cdf(value)
        sample = self.lognormal.sample(self.shape)
        return cdf, sample
```

调用网络
```python
shape = (2, 3)
net = Net(shape=shape, name="LogNormal")
x = np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32)
tx = Tensor(x, dtype=dtype.float32)
cdf, sample = net(tx)
print("cdf: ", cdf)
print("sample: ", sample)
```
输出为：
```python
cdf:  [0.7558914 0.8640314 0.9171715 0.9462397]
sample:  [[0.21036398 0.44932044 0.5669641 ]
 [1.4103683  6.724116   0.97894996]]
```

## 深度概率网络：mindspore.nn.probability.dpn

使用MindSpore深度概率编程库来构造变分自编码器（VAE）进行推理尤为简单。我们只需要自定义编码器和解码器（DNN模型），调用VAE或CVAE接口形成其派生网络，然后调用ELBO接口进行优化，最后使用SVI接口进行变分推理。这样做的好处是，不熟悉变分推理的用户可以像构建DNN模型一样来构建概率模型，而熟悉的用户可以调用这些接口来构建更为复杂的概率模型。VAE的接口在`mindspore.nn.probability.dpn`下面，dpn代表的是Deep probabilistic network，这里提供了一些基本的深度概率网络的接口，例如VAE。

### VAE

首先，我们需要先自定义encoder和decoder，调用`mindspore.nn.probability.dpn.VAE`接口来构建VAE网络，我们除了传入encoder和decoder之外，还需要传入encoder输出变量的维度hidden size，以及VAE网络存储潜在变量的维度latent size，一般latent size会小于hidden size。

```python
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.nn.probability.dpn import VAE

IMAGE_SHAPE = (-1, 1, 32, 32)


class Encoder(nn.Cell):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Dense(1024, 800)
        self.fc2 = nn.Dense(800, 400)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


class Decoder(nn.Cell):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Dense(400, 1024)
        self.sigmoid = nn.Sigmoid()
        self.reshape = P.Reshape()

    def construct(self, z):
        z = self.fc1(z)
        z = self.reshape(z, IMAGE_SHAPE)
        z = self.sigmoid(z)
        return z


encoder = Encoder()
decoder = Decoder()
vae = VAE(encoder, decoder, hidden_size=400, latent_size=20)
```
### ConditionalVAE

类似地，ConditionalVAE与VAE的使用方法比较相近，不同的是，ConditionalVAE利用了数据集的标签信息，属于有监督学习算法，其生成效果一般会比VAE好。

首先，先自定义encoder和decoder，并调用`mindspore.nn.probability.dpn.ConditionalVAE`接口来构建ConditionalVAE网络，这里的encoder和VAE的不同，因为需要传入数据集的标签信息；decoder和上述的一样。ConditionalVAE接口的传入则还需要传入数据集的标签类别个数，其余和VAE接口一样。

```
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.nn.probability.dpn import ConditionalVAE

IMAGE_SHAPE = (-1, 1, 32, 32)


class Encoder(nn.Cell):
    def __init__(self, num_classes):
        super(Encoder, self).__init__()
        self.fc1 = nn.Dense(1024 + num_classes, 400)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.concat = P.Concat(axis=1)
        self.one_hot = nn.OneHot(depth=num_classes)

    def construct(self, x, y):
        x = self.flatten(x)
        y = self.one_hot(y)
        input_x = self.concat((x, y))
        input_x = self.fc1(input_x)
        input_x = self.relu(input_x)
        return input_x


class Decoder(nn.Cell):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Dense(400, 1024)
        self.sigmoid = nn.Sigmoid()
        self.reshape = P.Reshape()

    def construct(self, z):
        z = self.fc1(z)
        z = self.reshape(z, IMAGE_SHAPE)
        z = self.sigmoid(z)
        return z


encoder = Encoder(num_classes=10)
decoder = Decoder()
cvae = ConditionalVAE(encoder, decoder, hidden_size=400, latent_size=20, num_classes=10)
```

加载数据集，我们可以使用Mnist数据集，具体的数据加载和预处理过程可以参考这里[Implementing an Image Classification Application](https://www.mindspore.cn/tutorial/en/master/quick_start/quick_start.html)，这里会用到create_dataset函数创建数据迭代器。

```python
ds_train = create_dataset(image_path, 128, 1)
```
接下来，需要用到infer接口进行VAE网络的变分推断。

## 概率推断算法：mindspore.nn.probability.infer

调用ELBO接口(`mindspore.nn.probability.infer.ELBO`)来定义VAE网络的损失函数，调用`WithLossCell`封装VAE网络和损失函数，并定义优化器，之后传入SVI接口(`mindspore.nn.probability.infer.SVI`)。SVI的`run`函数可理解为VAE网络的训练，可以指定训练的`epochs`，返回结果为训练好的网络；`get_train_loss`函数可以返回训练好后模型的loss。

```python
from mindspore.nn.probability.infer import ELBO, SVI

net_loss = ELBO(latent_prior='Normal', output_prior='Normal')
net_with_loss = nn.WithLossCell(vae, net_loss)
optimizer = nn.Adam(params=vae.trainable_params(), learning_rate=0.001)

vi = SVI(net_with_loss=net_with_loss, optimizer=optimizer)
vae = vi.run(train_dataset=ds_train, epochs=10)
trained_loss = vi.get_train_loss()
```
最后，得到训练好的VAE网络后，我们可以使用`vae.generate_sample`生成新样本，需要传入待生成样本的个数，及生成样本的shape，shape需要保持和原数据集中的样本shape一样；当然，我们也可以使用`vae.reconstruct_sample`重构原来数据集中的样本，来测试VAE网络的重建能力。
```python
generated_sample = vae.generate_sample(64, IMAGE_SHAPE)
for sample in ds_train.create_dict_iterator():
    sample_x = Tensor(sample['image'], dtype=mstype.float32)
    reconstructed_sample = vae.reconstruct_sample(sample_x)
print('The shape of the generated sample is ', generated_sample.shape)
```
我们可以看一下新生成样本的shape：
```
The shape of the generated sample is  (64, 1, 32, 32)
```
ConditionalVAE训练过程和vae的过程类似，但需要注意的是使用训练好的ConditionalVAE网络生成新样本和重建新样本时，需要输入标签信息，例如下面生成的新样本就是64个0-7的数字。

```
sample_label = Tensor([i for i in range(0, 8)] * 8, dtype=mstype.int32)
generated_sample = cvae.generate_sample(sample_label, 64, IMAGE_SHAPE)
for sample in ds_train.create_dict_iterator():
    sample_x = Tensor(sample['image'], dtype=mstype.float32)
    sample_y = Tensor(sample['label'], dtype=mstype.int32)
    reconstructed_sample = cvae.reconstruct_sample(sample_x, sample_y)
print('The shape of the generated sample is ', generated_sample.shape)
```
查看一下新生成的样本的shape：
```
The shape of the generated sample is  (64, 1, 32, 32)
```

如果希望新生成的样本更好，更清晰，用户可以自己定义更复杂的encoder和decoder，这里的示例只用了两层全连接层，仅供示例的指导。

## 贝叶斯层：mindspore.nn.probability.bnn_layers

下面的范例使用MindSpore的`nn.probability.bnn_layers`中的API实现BNN图片分类模型。MindSpore的`nn.probability.bnn_layers`中的API包括`NormalPrior`，`NormalPosterior`，`ConvReparam`，`DenseReparam`和`WithBNNLossCell`。

- ### 处理数据

```
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.transforms.vision import Inter
from mindspore.common import dtype as mstype


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """
    create dataset for train or test
    """
    # 定义数据集
    mnist_ds = ds.MnistDataset(data_path)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 定义映射操作
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # 对图像应用映射操作
    mnist_ds = mnist_ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=resize_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_nml_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=hwc2chw_op, num_parallel_workers=num_parallel_workers)

    # 应用DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds

```
### 定义模型

利用bnn_layers构建贝叶斯神经网络的方法与构建普通的神经网络相同。值得注意的是，bnn_layers和普通的神经网络层可以互相组合。当网络中包含bnn_layers时，需要将`WithLossCel`l替换为适用于BNN的`WithBNNLossCell`。除了`backbone`和`loss_fn`两个参数之外，`WithBNNLossCell`增加了`dnn_factor`和`bnn_factor`两个参数。`dnn_factor`是由损失函数计算得到的网络整体损失的系数，`bnn_factor`是每个贝叶斯层的KL散度的系数，这两个参数是用来平衡网络整体损失和贝叶斯层的KL散度的，防止KL散度的值过大掩盖了网络整体损失。

```
import mindspore.nn as nn
from mindspore.nn.probability import bnn_layers
from mindspore.ops import operations as P

class BNNLeNet5(nn.Cell):
    def __init__(self, num_class=10):
        super(BNNLeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = bnn_layers.ConvReparam(1, 6, 5, stride=1, padding=0, has_bias=False, pad_mode="valid")
        self.conv2 = bnn_layers.ConvReparam(6, 16, 5, stride=1, padding=0, has_bias=False, pad_mode="valid")
        self.fc1 = bnn_layers.DenseReparam(16 * 5 * 5, 120, activation=nn.ReLU())
        self.fc2 = bnn_layers.DenseReparam(120, 84)
        self.fc3 = bnn_layers.DenseReparam(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

network = BNNLeNet5()
# 定义损失函数
criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
# 定义优化器
optimizer = nn.AdamWeightDecay(params=network.trainable_params(), learning_rate=0.001)
net_with_loss = bnn_layers.WithBNNLossCell(network, criterion, dnn_factor=60000, bnn_factor=0.00001)
train_bnn_network = TrainOneStepCell(net_with_loss, optimizer)
train_bnn_network.set_train()
```
### 训练模型

```
def train_model(train_net, net, dataset):
    accs = []
    loss_sum = 0
    for i, data in enumerate(dataset.create_dict_iterator()):
        train_x = Tensor(data['image'].astype(np.float32))
        label = Tensor(data['label'].astype(np.int32))
        loss = train_net(train_x, label)
        output = net(train_x)
        log_output = P.LogSoftmax(axis=1)(output)
        acc = np.mean(log_output.asnumpy().argmax(axis=1) == label.asnumpy())
        accs.append(acc)
        loss_sum += loss.asnumpy()

    loss_sum = loss_sum / len(accs)
    acc_mean = np.mean(accs)
    return loss_sum, acc_mean


def validate_model(net, dataset):
    accs = []
    for _, data in enumerate(dataset.create_dict_iterator()):
        train_x = Tensor(data['image'].astype(np.float32))
        label = Tensor(data['label'].astype(np.int32))
        output = net(train_x)
        log_output = P.LogSoftmax(axis=1)(output)
        acc = np.mean(log_output.asnumpy().argmax(axis=1) == label.asnumpy())
        accs.append(acc)

    acc_mean = np.mean(accs)
    return acc_mean

train_set = create_dataset('./mnist_data/train', 64, 1)
test_set = create_dataset('./mnist_data/test', 64, 1)

epoch = 10

for i in range(epoch):
    train_loss, train_acc = train_model(train_bnn_network, network, train_set)

    valid_acc = validate_model(network, test_set)

    print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tvalidation Accuracy: {:.4f}'.format(
        i, train_loss, train_acc, valid_acc))
```


```
Epoch 0 	Training Loss 12925.5249 	Training Accuracy 0.9413 	validation Accuracy 0.9279
Epoch 1 	Training Loss 5287.0440 	Training Accuracy 0.9771 	validation Accuracy 0.9592
Epoch 2 	Training Loss 4186.9598 	Training Accuracy 0.9821 	validation Accuracy 0.9786
Epoch 3 	Training Loss 3538.2411 	Training Accuracy 0.9857 	validation Accuracy 0.9696
Epoch 4 	Training Loss 3255.8431 	Training Accuracy 0.9869 	validation Accuracy 0.9756
Epoch 5 	Training Loss 2880.1353 	Training Accuracy 0.9880 	validation Accuracy 0.9792
Epoch 6 	Training Loss 2569.0690 	Training Accuracy 0.9897 	validation Accuracy 0.9802
Epoch 7 	Training Loss 2391.5061 	Training Accuracy 0.9906 	validation Accuracy 0.9825
Epoch 8 	Training Loss 2165.1392 	Training Accuracy 0.9912 	validation Accuracy 0.9794
Epoch 9 	Training Loss 1910.2262 	Training Accuracy 0.9931 	validation Accuracy 0.9833
```

## 贝叶斯转换：mindspore.nn.probability.transform

对于不熟悉贝叶斯模型的DNN研究人员，MDP提供了高级API`TransformToBNN`，支持DNN模型一键转换成BNN模型。

### 定义DNN模型

本例使用的DNN模型是LeNet。

```
from mindspore.common.initializer import TruncatedNormal
import mindspore.nn as nn
import mindspore.ops.operations as P

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Num classes. Default: 10.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet5(num_class=10)

    """
    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
```
LeNet网络结构如下：

```
LeNet5
  (conv1) Conv2dinput_channels=1, output_channels=6, kernel_size=(5, 5),stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False
  (conv2) Conv2dinput_channels=6, output_channels=16, kernel_size=(5, 5),stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False
  (fc1) Densein_channels=400, out_channels=120, weight=Parameter (name=fc1.weight), has_bias=True, bias=Parameter (name=fc1.bias)
  (fc2) Densein_channels=120, out_channels=84, weight=Parameter (name=fc2.weight), has_bias=True, bias=Parameter (name=fc2.bias)
  (fc3) Densein_channels=84, out_channels=10, weight=Parameter (name=fc3.weight), has_bias=True, bias=Parameter (name=fc3.bias)
  (relu) ReLU
  (max_pool2d) MaxPool2dkernel_size=2, stride=2, pad_mode=VALID
  (flatten) Flatten
  
```

### 定义损失函数和优化器

接下来需要定义损失函数（Loss）和优化器（Optimizer）。本例使用交叉熵损失作为损失函数，Adam作为优化器。

```
network = LeNet5()

# loss function definition
criterion = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

# optimization definition
optimizer = AdamWeightDecay(params=network.trainable_params(), learning_rate=0.0001)

net_with_loss = WithLossCell(network, criterion)
train_network = TrainOneStepCell(net_with_loss, optimizer)
```
### 实例化TransformToBNN

`TransformToBNN`的`__init__`函数定义如下：

```
class TransformToBNN:
    def __init__(self, trainable_dnn, dnn_factor=1, bnn_factor=1):
        net_with_loss = trainable_dnn.network
        self.optimizer = trainable_dnn.optimizer
        self.backbone = net_with_loss.backbone_network
        self.loss_fn = getattr(net_with_loss, "_loss_fn")
        self.dnn_factor = dnn_factor
        self.bnn_factor = bnn_factor
        self.bnn_loss_file = None
```
参数`trainable_bnn`是经过`TrainOneStepCell`包装的可训练DNN模型，`dnn_factor`和`bnn_factor`分别为由损失函数计算得到的网络整体损失的系数和每个贝叶斯层的KL散度的系数。
MindSpore中实例化`TransformToBNN`的代码如下：

```
from mindspore.nn.probability import transforms

bnn_transformer = transforms.TransformToBNN(train_network, 60000, 0.000001)
```
#### 功能一：转换整个模型
`transform_to_bnn_model`方法可以将整个DNN模型转换为BNN模型。其定义如下

```
    def transform_to_bnn_model(self,
                               get_dense_args=lambda dp: {"in_channels": dp.in_channels, "has_bias": dp.has_bias,
                                                          "out_channels": dp.out_channels, "activation": dp.activation},
                               get_conv_args=lambda dp: {"in_channels": dp.in_channels, "out_channels": dp.out_channels,
                                                         "pad_mode": dp.pad_mode, "kernel_size": dp.kernel_size,
                                                         "stride": dp.stride, "has_bias": dp.has_bias,
                                                         "padding": dp.padding, "dilation": dp.dilation,
                                                         "group": dp.group},
                               add_dense_args=None,
                               add_conv_args=None):
        r"""
        Transform the whole DNN model to BNN model, and wrap BNN model by TrainOneStepCell.

        Args:
            get_dense_args (function): The arguments gotten from the DNN full connection layer. Default: lambda dp:
                {"in_channels": dp.in_channels, "out_channels": dp.out_channels, "has_bias": dp.has_bias}.
            get_conv_args (function): The arguments gotten from the DNN convolutional layer. Default: lambda dp:
                {"in_channels": dp.in_channels, "out_channels": dp.out_channels, "pad_mode": dp.pad_mode,
                "kernel_size": dp.kernel_size, "stride": dp.stride, "has_bias": dp.has_bias}.
            add_dense_args (dict): The new arguments added to BNN full connection layer. Default: {}.
            add_conv_args (dict): The new arguments added to BNN convolutional layer. Default: {}.

        Returns:
            Cell, a trainable BNN model wrapped by TrainOneStepCell.
       """
       
```
参数`get_dense_args`指定从DNN模型的全连接层中获取哪些参数，`get_conv_args`指定从DNN模型的卷积层中获取哪些参数，参数`add_dense_args`和`add_conv_args`分别指定了要为BNN层指定哪些新的参数值。需要注意的是，`add_dense_args`中的参数不能与`get_dense_args`重复，`add_conv_args`和`get_conv_args`也是如此。
在MindSpore中将整个DNN模型转换成BNN模型的代码如下：

```
train_bnn_network = bnn_transformer.transform_to_bnn_model()
```
整个模型转换后的结构如下：

```
LeNet5
  (conv1) ConvReparam
    in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, weight_mean=Parameter (name=conv1.weight_posterior.mean), weight_std=Parameter (name=conv1.weight_posterior.untransformed_std), has_bias=False
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1
      
    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None
      
    
  (conv2) ConvReparam
    in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, weight_mean=Parameter (name=conv2.weight_posterior.mean), weight_std=Parameter (name=conv2.weight_posterior.untransformed_std), has_bias=False
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1
      
    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None
      
    
  (fc1) DenseReparam
    in_channels=400, out_channels=120, weight_mean=Parameter (name=fc1.weight_posterior.mean), weight_std=Parameter (name=fc1.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc1.bias_posterior.mean), bias_std=Parameter (name=fc1.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1
      
    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None
      
    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1
      
    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None
      
    
  (fc2) DenseReparam
    in_channels=120, out_channels=84, weight_mean=Parameter (name=fc2.weight_posterior.mean), weight_std=Parameter (name=fc2.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc2.bias_posterior.mean), bias_std=Parameter (name=fc2.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1
      
    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None
      
    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1
      
    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None
      
    
  (fc3) DenseReparam
    in_channels=84, out_channels=10, weight_mean=Parameter (name=fc3.weight_posterior.mean), weight_std=Parameter (name=fc3.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc3.bias_posterior.mean), bias_std=Parameter (name=fc3.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1
      
    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None
      
    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1
      
    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None
      
    
  (relu) ReLU
  (max_pool2d) MaxPool2dkernel_size=2, stride=2, pad_mode=VALID
  (flatten) Flatten
  
```

#### 功能二：转换指定类型的层
`transform_to_bnn_layer`方法可以将DNN模型中指定类型的层（`nn.Dense`或者`nn.Conv2d`）转换为对应的贝叶斯层。其定义如下

```
 def transform_to_bnn_layer(self, dnn_layer, bnn_layer, get_args=None, add_args=None):
        r"""
        Transform a specific type of layers in DNN model to corresponding BNN layer.

        Args:
            dnn_layer_type (Cell): The type of DNN layer to be transformed to BNN layer. The optional values are
            nn.Dense, nn.Conv2d.
            bnn_layer_type (Cell): The type of BNN layer to be transformed to. The optional values are
                DenseReparameterization, ConvReparameterization.
            get_args (dict): The arguments gotten from the DNN layer. Default: None.
            add_args (dict): The new arguments added to BNN layer. Default: None.

        Returns:
            Cell, a trainable model wrapped by TrainOneStepCell, whose sprcific type of layer is transformed to the corresponding bayesian layer.
        """        
```
参数`dnn_layer`指定将哪个类型的DNN层转换成BNN层，`bnn_layer`指定DNN层将转换成哪个类型的BNN层，`get_args`和`add_args`分别指定从DNN层中获取哪些参数和要为BNN层的哪些参数重新赋值。
在MindSpore中将DNN模型中的Dense层转换成相应贝叶斯层`DenseReparam`的代码如下：

```
train_bnn_network = bnn_transformer.transform_to_bnn_layer(nn.Dense, bnn_layers.DenseReparam)
```
转换后的结构如下：

```
LeNet5
  (conv1) Conv2dinput_channels=1, output_channels=6, kernel_size=(5, 5),stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False
  (conv2) Conv2dinput_channels=6, output_channels=16, kernel_size=(5, 5),stride=(1, 1),  pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False
  (fc1) DenseReparam
    in_channels=400, out_channels=120, weight_mean=Parameter (name=fc1.weight_posterior.mean), weight_std=Parameter (name=fc1.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc1.bias_posterior.mean), bias_std=Parameter (name=fc1.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1
      
    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None
      
    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1
      
    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None
      
    
  (fc2) DenseReparam
    in_channels=120, out_channels=84, weight_mean=Parameter (name=fc2.weight_posterior.mean), weight_std=Parameter (name=fc2.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc2.bias_posterior.mean), bias_std=Parameter (name=fc2.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1
      
    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None
      
    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1
      
    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None
      
    
  (fc3) DenseReparam
    in_channels=84, out_channels=10, weight_mean=Parameter (name=fc3.weight_posterior.mean), weight_std=Parameter (name=fc3.weight_posterior.untransformed_std), has_bias=True, bias_mean=Parameter (name=fc3.bias_posterior.mean), bias_std=Parameter (name=fc3.bias_posterior.untransformed_std)
    (weight_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1
      
    (weight_posterior) NormalPosterior
      (normal) Normalbatch_shape = None
      
    (bias_prior) NormalPrior
      (normal) Normalmean = 0.0, standard deviation = 0.1
      
    (bias_posterior) NormalPosterior
      (normal) Normalbatch_shape = None
      
    
  (relu) ReLU
  (max_pool2d) MaxPool2dkernel_size=2, stride=2, pad_mode=VALID
  (flatten) Flatten
  
```

## 贝叶斯工具箱：mindspore.nn.probability.toolbox

贝叶斯神经网络的优势之一就是可以获取不确定性，MDP在上层提供了不确定性估计的工具箱，用户可以很方便地使用该工具箱计算不确定性。不确定性意味着深度学习模型不知道什么。目前，大多数深度学习算法只能给出高置信度的预测结果，而不能判断预测结果的确定性，不确定性主要有两种类型：偶然不确定性和认知不确定性。

- 偶然不确定性（Aleatoric Uncertainty）：描述数据中的内在噪声，即无法避免的误差，这个现象不能通过增加采样数据来削弱。
- 认知不确定性（Epistemic Uncertainty）：模型自身对输入数据的估计可能因为训练不佳、训练数据不够等原因而不准确，可以通过增加训练数据等方式来缓解。

不确定性评估工具箱的接口如下：
- `model`: 待评估不确定性的已训练好的模型。
- `train_dataset`: 用于训练的数据集，迭代器类型。
- `task_type`: 模型的类型，字符串，输入“regression”或者“classification”。
- `num_classes`: 如果是分类模型，需要指定类别的标签数量。
- `epochs`: 用于训练不确定模型的迭代数。
- `epi_uncer_model_path`: 用于存储或加载计算认知不确定性的模型的路径。
- `ale_uncer_model_path`: 用于存储或加载计算偶然不确定性的模型的路径。
- `save_model`: 布尔类型，是否需要存储模型。

在使用前，需要先训练好模型，以LeNet5为例，使用方式如下：
```
from mindspore.nn.probability.toolbox.uncertainty_evaluation import UncertaintyEvaluation
from mindspore.train.serialization import load_checkpoint, load_param_into_net

if __name__ == '__main__':
    # get trained model
    network = LeNet5()
    param_dict = load_checkpoint('checkpoint_lenet.ckpt')
    load_param_into_net(network, param_dict)
    # get train and eval dataset
    ds_train = create_dataset('workspace/mnist/train')
    ds_eval = create_dataset('workspace/mnist/test')
    evaluation = UncertaintyEvaluation(model=network,
                                       train_dataset=ds_train,
                                       task_type='classification',
                                       num_classes=10,
                                       epochs=1,
                                       epi_uncer_model_path=None,
                                       ale_uncer_model_path=None,
                                       save_model=False)
    for eval_data in ds_eval.create_dict_iterator():
        eval_data = Tensor(eval_data['image'], mstype.float32)
        epistemic_uncertainty = evaluation.eval_epistemic_uncertainty(eval_data)
        aleatoric_uncertainty = evaluation.eval_aleatoric_uncertainty(eval_data)
    print('The shape of epistemic uncertainty is ', epistemic_uncertainty.shape)
    print('The shape of epistemic uncertainty is ', aleatoric_uncertainty.shape)
```
`eval_epistemic_uncertainty`计算的是认知不确定性，也叫模型不确定性，对于每一个样本的每个预测标签都会有一个不确定值；`eval_aleatoric_uncertainty`计算的是偶然不确定性，也叫数据不确定性，对于每一个样本都会有一个不确定值。
所以输出为：

```
The shape of epistemic uncertainty is (32, 10)
The shape of epistemic uncertainty is (32,)
```