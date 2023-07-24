# 深度概率编程库

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/programming_guide/source_zh_cn/probability.md)

MindSpore深度概率编程的目标是将深度学习和贝叶斯学习结合，包括概率分布、概率分布映射、深度概率网络、概率推断算法、贝叶斯层、贝叶斯转换和贝叶斯工具箱，面向不同的开发者。对于专业的贝叶斯学习用户，提供概率采样、推理算法和模型构建库；另一方面，为不熟悉贝叶斯深度学习的用户提供了高级的API，从而不用更改深度学习编程逻辑，即可利用贝叶斯模型。

## 概率分布

概率分布（`mindspore.nn.probability.distribution`）是概率编程的基础。`Distribution` 类提供多样的概率统计接口，例如概率密度函数 *pdf* 、累积密度函数 *cdf* 、散度计算 *kl_loss* 、抽样 *sample* 等。现有的概率分布实例包括高斯分布，伯努利分布，指数型分布，几何分布和均匀分布。

### 概率分布类

- `Distribution`：所有概率分布的基类。

- `Bernoulli`：伯努利分布。参数为试验成功的概率。

- `Exponential`：指数型分布。参数为率参数。

- `Geometric`：几何分布。参数为一次伯努利试验成功的概率。

- `Normal`：正态（高斯）分布。参数为均值和标准差。

- `Uniform`：均匀分布。参数为数轴上的最小值和最大值。

- `Categorical`：类别分布。每种类别出现的概率。

- `LogNormal`：对数正态分布。参数为位置参数和规模参数。

- `Gumbel`: 耿贝尔极值分布。参数为位置参数和规模参数。

- `Logistic`：逻辑斯谛分布。参数为位置参数和规模参数。

- `Cauchy`：柯西分布。参数为位置参数和规模参数。

#### Distribution基类

`Distribution` 是所有概率分布的基类。

接口介绍：`Distribution` 类支持的函数包括 `prob`、`log_prob`、`cdf`、`log_cdf`、`survival_function`、`log_survival`、`mean`、`sd`、`var`、`entropy`、`kl_loss`、`cross_entropy` 和 `sample` 。分布不同，所需传入的参数也不同。只有在派生类中才能使用，由派生类的函数实现决定参数。

- `prob` ：概率密度函数（PDF）/ 概率质量函数（PMF）。
- `log_prob` ：对数似然函数。
- `cdf` ：累积分布函数（CDF）。
- `log_cdf` ：对数累积分布函数。
- `survival_function` ：生存函数。
- `log_survival` ：对数生存函数。
- `mean` ：均值。
- `sd` ：标准差。
- `var` ：方差。
- `entropy` ：熵。
- `kl_loss` ：Kullback-Leibler 散度。
- `cross_entropy` ：两个概率分布的交叉熵。
- `sample` ：概率分布的随机抽样。
- `get_dist_args` ：概率分布在网络中使用的参数。
- `get_dist_type` ：概率分布的类型。

#### 伯努利分布(Bernoulli)

伯努利分布，继承自 `Distribution` 类。

属性:

- `Bernoulli.probs`：返回伯努利试验成功的概率，类型为`Tensor`。

`Distribution` 基类调用 `Bernoulli` 中私有接口以实现基类中的公有接口。`Bernoulli` 支持的公有接口为：

- `mean`，`mode`，`var`，`sd`：可选择传入 试验成功的概率 *probs1* 。
- `entropy`：可选择传入 试验成功的概率 *probs1* 。
- `cross_entropy`，`kl_loss`：必须传入 *dist* 和 *probs1_b* 。*dist* 为另一分布的类型，目前只支持此处为 *‘Bernoulli’* 。 *probs1_b* 为分布 *b* 的试验成功概率。可选择传入分布 *a* 的参数 *probs1_a* 。
- `prob`，`log_prob`，`cdf`，`log_cdf`，`survival_function`，`log_survival`：必须传入 *value* 。可选择传入试验成功的概率 *probs* 。
- `sample`：可选择传入样本形状 *shape* 和试验成功的概率 *probs1* 。
- `get_dist_args` ：可选择传入试验成功的概率 *probs*。返回值为`(probs,)`，类型为tuple。
- `get_dist_type` ：返回 *‘Bernoulli’* 。

#### 指数分布(Exponential)

指数分布，继承自 `Distribution` 类。

属性:

- `Exponential.rate`：返回分布的率参数，类型为`Tensor`。

`Distribution` 基类调用 `Exponential` 私有接口以实现基类中的公有接口。`Exponential` 支持的公有接口为：

- `mean`，`mode`，`var`，`sd`：可选择传入率参数 *rate* 。
- `entropy`：可选择传入率参数 *rate* 。
- `cross_entropy`，`kl_loss`：必须传入 *dist* 和 *rate_b* 。 *dist* 为另一分布的类型的名称， 目前只支持此处为 *‘Exponential’* 。*rate_b* 为分布 *b* 的率参数。可选择传入分布 *a* 的参数 *rate_a* 。
- `prob`，`log_prob`，`cdf`，`log_cdf`，`survival_function`，`log_survival`：必须传入 *value* 。可选择传入率参数 *rate* 。
- `sample`：可选择传入样本形状 *shape* 和率参数 *rate* 。
- `get_dist_args` ：可选择传入率参数 *rate* 。返回值为`(rate,)`，类型为tuple。
- `get_dist_type` ：返回 *‘Exponential’* 。

#### 几何分布(Geometric)

几何分布，继承自 `Distribution` 类。

属性:

- `Geometric.probs`：返回伯努利试验成功的概率，类型为`Tensor`。

`Distribution` 基类调用 `Geometric` 中私有接口以实现基类中的公有接口。`Geometric` 支持的公有接口为：

- `mean`，`mode`，`var`，`sd`：可选择传入试验成功的概率 *probs1* 。
- `entropy`：可选择传入 试验成功的概率 *probs1* 。
- `cross_entropy`，`kl_loss`：必须传入 *dist* 和 *probs1_b* 。*dist* 为另一分布的类型的名称，目前只支持此处为 *‘Geometric’* 。 *probs1_b* 为分布 *b* 的试验成功概率。可选择传入分布 *a* 的参数 *probs1_a* 。
- `prob`，`log_prob`，`cdf`，`log_cdf`，`survival_function`，`log_survival`：必须传入 *value* 。可选择传入试验成功的概率 *probs1* 。
- `sample`：可选择传入样本形状 *shape* 和试验成功的概率 *probs1* 。
- `get_dist_args` ：可选择传入试验成功的概率 *probs1* 。返回值为`(probs1,)`，类型为tuple。
- `get_dist_type` ：返回 *‘Geometric’* 。

#### 正态分布(Normal)

正态（高斯）分布，继承自 `Distribution` 类。

`Distribution` 基类调用 `Normal` 中私有接口以实现基类中的公有接口。`Normal` 支持的公有接口为：

- `mean`，`mode`，`var`，`sd`：可选择传入分布的参数均值 *mean* 和标准差 *sd* 。
- `entropy`：可选择传入分布的参数均值 *mean* 和标准差 *sd* 。
- `cross_entropy`，`kl_loss`：必须传入 *dist* ，*mean_b* 和 *sd_b* 。*dist* 为另一分布的类型的名称，目前只支持此处为 *‘Normal’* 。*mean_b* 和 *sd_b* 为分布 *b* 的均值和标准差。可选择传入分布的参数 *a* 均值 *mean_a* 和标准差 *sd_a* 。
- `prob`，`log_prob`，`cdf`，`log_cdf`，`survival_function`，`log_survival`：必须传入 *value* 。可选择分布的参数包括均值 *mean_a* 和标准差 *sd_a* 。
- `sample`：可选择传入样本形状 *shape* 和分布的参数包括均值 *mean_a* 和标准差 *sd_a* 。
- `get_dist_args` ：可选择传入分布的参数均值 *mean* 和标准差 *sd* 。返回值为`(mean, sd)`，类型为tuple。
- `get_dist_type` ：返回 *‘Normal’* 。

#### 均匀分布(Uniform)

均匀分布，继承自 `Distribution` 类。

属性:

- `Uniform.low`：返回分布的最小值，类型为`Tensor`。
- `Uniform.high`：返回分布的最大值，类型为`Tensor`。

`Distribution` 基类调用 `Uniform` 以实现基类中的公有接口。`Uniform` 支持的公有接口为：

- `mean`，`mode`，`var`，`sd`：可选择传入分布的参数最大值 *high* 和最小值 *low* 。
- `entropy`：可选择传入分布的参数最大值 *high* 和最小值 *low* 。
- `cross_entropy`，`kl_loss`：必须传入 *dist* ，*high_b* 和 *low_b* 。*dist* 为另一分布的类型的名称，目前只支持此处为 *‘Uniform’* 。 *high_b* 和 *low_b* 为分布 *b* 的参数。可选择传入分布 *a* 的参数即最大值 *high_a* 和最小值 *low_a* 。
- `prob`，`log_prob`，`cdf`，`log_cdf`，`survival_function`，`log_survival`：必须传入 *value* 。可选择传入分布的参数最大值 *high* 和最小值 *low* 。
- `sample`：可选择传入 *shape* 和分布的参数即最大值 *high* 和最小值 *low* 。
- `get_dist_args` ：可选择传入分布的参数最大值 *high* 和最小值 *low* 。返回值为`(low, high)`，类型为tuple。
- `get_dist_type` ：返回 *‘Uniform’* 。

#### 多类别分布（Categorical）

多类别分布，继承自 `Distribution` 类。

属性:

- `Categorical.probs`：返回各种类别的概率，类型为`Tensor`。

`Distribution` 基类调用 `Categorical` 以实现基类中的公有接口。`Categorical` 支持的公有接口为：

- `mean`，`mode`，`var`，`sd`：可选择传入分布的参数类别概率 *probs*。
- `entropy`：可选择传入分布的参数类别概率 *probs* 。
- `cross_entropy`，`kl_loss`：必须传入 *dist* ，*probs_b* 。*dist* 为另一分布的类型的名称，目前只支持此处为 *‘Categorical’* 。 *probs_b* 为分布 *b* 的参数。可选择传入分布 *a* 的参数即 *probs_a* 。
- `prob`，`log_prob`，`cdf`，`log_cdf`，`survival_function`，`log_survival`：必须传入 *value* 。可选择传入分布的参数类别概率 *probs* 。
- `sample`：可选择传入 *shape* 和类别概率 *probs* 。
- `get_dist_args` ：可选择传入分布的参数类别概率 *probs* 。返回值为`(probs,)`，类型为tuple。
- `get_dist_type` ：返回 *‘Categorical’* 。

#### 对数正态分布(LogNormal)

对数正态分布，继承自 `TransformedDistribution` 类，由 `Exp` Bijector 和 `Normal` Distribution 构成。

属性：

- `LogNormal.loc`：返回分布的位置参数，类型为`Tensor`。
- `LogNormal.scale`：返回分布的规模参数，类型为`Tensor`。

`Distribution` 基类调用 `LogNormal`及 `TransformedDistribution` 中私有接口以实现基类中的公有接口。`LogNormal` 支持的公有接口为：

- `mean`，`mode`，`var`，`sd`：可选择传入分布的位置参数*loc*和规模参数*scale* 。
- `entropy`：可选择传入分布的位置参数 *loc* 和规模参数 *scale* 。
- `cross_entropy`，`kl_loss`：必须传入 *dist* ，*loc_b* 和 *scale_b* 。*dist* 为另一分布的类型的名称，目前只支持此处为 *‘LogNormal’* 。*loc_b* 和 *scale_b* 为分布 *b* 的均值和标准差。可选择传入分布的参数 *a* 均值 *loc_a* 和标准差 *sclae_a* 。
- `prob`，`log_prob`，`cdf`，`log_cdf`，`survival_function`，`log_survival`：必须传入 *value* 。可选择分布的参数包括均值 *loc_a* 和标准差 *scale_a* 。`Distribution` 基类调用 `TransformedDistribution`私有接口。
- `sample`：可选择传入样本形状 *shape* 和分布的参数包括均值 *loc_a* 和标准差 *scale_a* 。`Distribution` 基类调用 `TransformedDistribution`私有接口。
- `get_dist_args` ：可选择传入分布的位置参数 *loc* 和规模参数*scale* 。返回值为`(loc, scale)`，类型为tuple。
- `get_dist_type` ：返回 *‘LogNormal’* 。

#### 柯西分布(Cauchy)

柯西分布，继承自 `Distribution` 类。

属性：

- `Cauchy.loc`：返回分布的位置参数，类型为`Tensor`。
- `Cauchy.scale`：返回分布的规模参数，类型为`Tensor`。

`Distribution` 基类调用 `Cauchy` 中私有接口以实现基类中的公有接口。`Cauchy` 支持的公有接口为：

- `entropy`：可选择传入分布的位置参数*loc*和规模参数*scale*。
- `cross_entropy`，`kl_loss`：必须传入 *dist* ，*loc_b* 和 *scale_b* 。*dist* 为另一分布的类型的名称，目前只支持此处为 *‘Cauchy’* 。*loc_b* 和 *scale_b* 为分布 *b* 的位置参数和规模参数。可选择传入分布的参数 *a* 位置 *loc_a* 和规模 *scale_a* 。
- `prob`，`log_prob`，`cdf`，`log_cdf`，`survival_function`，`log_survival`：必须传入 *value* 。可选择传入分布的位置参数 *loc* 和规模参数 *scale* 。
- `sample`：可选择传入样本形状 *shape* 和分布的参数包括分布的位置参数 *loc* 和规模参数 *scale* 。
- `get_dist_args` ：可选择传入分布的位置参数 *loc* 和规模参数 *scale* 。返回值为`(loc, scale)`，类型为tuple。
- `get_dist_type` ：返回 *‘Cauchy’* 。

#### 耿贝尔极值分布(Gumbel)

耿贝尔极值分布，继承自 `TransformedDistribution` 类，由 `GumbelCDF` Bijector和 `Uniform` Distribution 构成。

属性：

- `Gumbel.loc`：返回分布的位置参数，类型为`Tensor`。
- `Gumbel.scale`：返回分布的规模参数，类型为`Tensor`。

`Distribution` 基类调用 `Gumbel` 中私有接口以实现基类中的公有接口。`Gumbel` 支持的公有接口为：

- `mean`，`mode`，`var`，`sd`：无参数 。
- `entropy`：无参数 。
- `cross_entropy`，`kl_loss`：必须传入 *dist* ，*loc_b* 和 *scale_b* 。*dist* 为另一分布的类型的名称，目前只支持此处为 *‘Gumbel’* 。*loc_b* 和 *scale_b* 为分布 *b* 的位置参数和规模参数。
- `prob`，`log_prob`，`cdf`，`log_cdf`，`survival_function`，`log_survival`：必须传入 *value* 。
- `sample`：可选择传入样本形状 *shape* 。
- `get_dist_args` ：可选择传入分布的位置参数 *loc* 和规模参数 *scale* 。返回值为`(loc, scale)`，类型为tuple。
- `get_dist_type` ：返回 *‘Gumbel’* 。

#### 逻辑斯谛分布(Logistic)

逻辑斯谛分布，继承自 `Distribution` 类。

属性：

- `Logistic.loc`：返回分布的位置参数，类型为`Tensor`。
- `Logistic.scale`：返回分布的规模参数，类型为`Tensor`。

`Distribution` 基类调用 `Logistic` 中私有接口以实现基类中的公有接口。`Logistic` 支持的公有接口为：

- `mean`，`mode`，`var`，`sd`：可选择传入分布的位置参数 *loc* 和规模参数 *scale* 。
- `entropy`：可选择传入分布的位置参数 *loc* 和规模参数 *scale* 。
- `prob`，`log_prob`，`cdf`，`log_cdf`，`survival_function`，`log_survival`：必须传入 *value* 。可选择传入分布的位置参数 *loc* 和规模参数 *scale* 。
- `sample`：可选择传入样本形状 *shape* 和分布的参数包括分布的位置参数 *loc* 和规模参数 *scale* 。
- `get_dist_args` ：可选择传入分布的位置参数 *loc* 和规模参数 *scale* 。返回值为`(loc, scale)`，类型为tuple。
- `get_dist_type` ：返回 *‘Logistic’* 。

#### 泊松分布(Poisson)

泊松分布，继承自 `Distribution` 类。

属性：

- `Poisson.rate`：返回分布的率参数，类型为`Tensor`。

`Distribution` 基类调用 `Poisson` 中私有接口以实现基类中的公有接口。`Poisson` 支持的公有接口为：

- `mean`，`mode`，`var`，`sd`：可选择传入分布的率参数 *rate* 。
- `prob`，`log_prob`，`cdf`，`log_cdf`，`survival_function`，`log_survival`：必须传入 *value* 。可选择传入分布的率参数 *rate* 。
- `sample`：可选择传入样本形状 *shape* 和分布的率参数 *rate* 。
- `get_dist_args` ：可选择传入分布的率参数 *rate* 。返回值为`(rate,)`，类型为tuple。
- `get_dist_type` ：返回 *‘Poisson’* 。

#### 伽马分布(Gamma)

伽马分布，继承自 `Distribution` 类。

属性：

- `Gamma.concentration`：返回分布的参数 `concentration` ，类型为`Tensor`。
- `Gamma.rate`：返回分布的参数 `rate` ，类型为`Tensor`。

`Distribution` 基类调用 `Gamma` 中私有接口以实现基类中的公有接口。`Gamma` 支持的公有接口为：

- `mean`，`mode`，`sd`，`var`：可选择传入分布的参数 *concentration* 和参数 *rate* 。
- `entropy`：可选择传入分布的参数 *concentration* 和参数 *rate* 。
- `prob`，`log_prob`，`cdf`，`log_cdf`，`survival_function`，`log_survival`：必须传入 *value* 。可选择传入分布的参数 *concentration* 和参数 *rate* 。
- `cross_entropy`，`kl_loss`：必须传入 *dist* ，*concentration_b* 和 *rate_b* 。*dist* 为另一分布的类型的名称，目前只支持此处为 *‘Gamma’* 。*concentration_b* 和 *rate_b* 为分布 *b* 的参数。可选择传入分布 *a* 的参数即 *concentration_a* 和 *rate_a* 。
- `sample`：可选择传入样本形状 *shape* 和分布的参数包括分布的参数 *concentration* 和参数 *rate* 。
- `get_dist_args` ：可选择传入分布的参数 *concentration* 和参数 *rate* 。返回值为`(concentration, rate)`，类型为tuple。
- `get_dist_type` ：返回 *‘Gamma’* 。

#### 贝塔分布(Beta)

贝塔分布，继承自 `Distribution` 类。

属性：

- `Beta.concentration1`：返回分布的参数 `concentration1` ，类型为`Tensor`。
- `Beta.concentration0`：返回分布的参数 `concentration0` ，类型为`Tensor`。

`Distribution` 基类调用 `Beta` 中私有接口以实现基类中的公有接口。`Beta` 支持的公有接口为：

- `mean`，`mode`，`sd`，`var`：可选择传入分布的参数 *concentration1* 和参数 *concentration0* 。
- `entropy`：可选择传入分布的参数 *concentration1* 和参数 *concentration0* 。
- `prob`，`log_prob`：必须传入 *value* 。可选择传入分布的参数 *concentration1* 和参数 *concentration0* 。
- `cross_entropy`，`kl_loss`：必须传入 *dist* ，*concentration1_b* 和 *concentration1_b* 。*dist* 为另一分布的类型的名称，目前只支持此处为 *‘Beta’* 。*concentration1_b* 和 *concentration1_b* 为分布 *b* 的参数。可选择传入分布 *a* 的参数即 *concentration1_a* 和 *concentration0_a* 。
- `sample`：可选择传入样本形状 *shape* 和分布的参数包括分布的位置参数 *loc* 和规模参数 *scale* 。
- `get_dist_args` ：可选择传入分布的参数 *concentration1* 和参数 *concentration0* 。返回值为`(concentration1, concentration0)`，类型为tuple。
- `get_dist_type` ：返回 *‘Beta’* 。

### 概率分布类在PyNative模式下的应用

`Distribution` 子类可在 **PyNative** 模式下使用。

以 `Normal` 为例， 创建一个均值为0.0、标准差为1.0的正态分布，然后计算相关函数。

```python
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.context as context
import mindspore.nn.probability.distribution as msd
context.set_context(mode=context.PYNATIVE_MODE)

my_normal = msd.Normal(0.0, 1.0, dtype=mstype.float32)

mean = my_normal.mean()
var = my_normal.var()
entropy = my_normal.entropy()

value = Tensor([-0.5, 0.0, 0.5], dtype=mstype.float32)
prob = my_normal.prob(value)
cdf = my_normal.cdf(value)

mean_b = Tensor(1.0, dtype=mstype.float32)
sd_b = Tensor(2.0, dtype=mstype.float32)
kl = my_normal.kl_loss('Normal', mean_b, sd_b)

# get the distribution args as a tuple
dist_arg = my_normal.get_dist_args()

print("mean: ", mean)
print("var: ", var)
print("entropy: ", entropy)
print("prob: ", prob)
print("cdf: ", cdf)
print("kl: ", kl)
print("dist_arg: ", dist_arg)
```

输出为：

```text
mean:  0.0
var:  1.0
entropy:  1.4189385
prob:  [0.35206532 0.3989423  0.35206532]
cdf:  [0.30853754 0.5        0.69146246]
kl:  0.44314718
dist_arg: (Tensor(shape=[], dtype=Float32, value= 0), Tensor(shape=[], dtype=Float32, value= 1))
```

### 概率分布类在图模式下的应用

在图模式下，`Distribution` 子类可用在网络中。

```python
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.context as context
import mindspore.nn.probability.distribution as msd
context.set_context(mode=context.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.normal = msd.Normal(0.0, 1.0, dtype=mstype.float32)

    def construct(self, value, mean, sd):
        pdf = self.normal.prob(value)
        kl = self.normal.kl_loss("Normal", mean, sd)
        return pdf, kl

net = Net()
value = Tensor([-0.5, 0.0, 0.5], dtype=mstype.float32)
mean = Tensor(1.0, dtype=mstype.float32)
sd = Tensor(1.0, dtype=mstype.float32)
pdf, kl = net(value, mean, sd)
print("pdf: ", pdf)
print("kl: ", kl)
```

输出为：

```text
pdf:  [0.35206532 0.3989423  0.35206532]
kl:  0.5
```

### TransformedDistribution类接口设计

`TransformedDistribution` 继承自 `Distribution` ，是可通过映射f(x)变化得到的数学分布的基类。其接口包括：

1. 属性

    - `bijector`：返回分布的变换方法。
    - `distribution`：返回原始分布。
    - `is_linear_transformation`：返回线性变换标志。

2. 接口函数（以下接口函数的参数与构造函数中 `distribution` 的对应接口的参数相同）。

    - `cdf`：累积分布函数（CDF）。
    - `log_cdf`：对数累积分布函数。
    - `survival_function`：生存函数。
    - `log_survival`：对数生存函数。
    - `prob`：概率密度函数（PDF）/ 概率质量函数（PMF）。
    - `log_prob`：对数似然函数。
    - `sample`：随机取样。
    - `mean`：无参数。只有当 `Bijector.is_constant_jacobian=true` 时可调用。

### PyNative模式下调用TransformedDistribution实例

`TransformedDistribution` 子类可在 **PyNative** 模式下使用。

这里构造一个 `TransformedDistribution` 实例，使用 `Normal` 分布作为需要变换的分布类，使用 `Exp` 作为映射变换，可以生成 `LogNormal` 分布。

```python
import numpy as np
import mindspore.nn as nn
import mindspore.nn.probability.bijector as msb
import mindspore.nn.probability.distribution as msd
import mindspore.context as context
from mindspore import Tensor, dtype

context.set_context(mode=context.PYNATIVE_MODE)

normal = msd.Normal(0.0, 1.0, dtype=dtype.float32)
exp = msb.Exp()
LogNormal = msd.TransformedDistribution(exp, normal, seed=0, name="LogNormal")

# compute cumulative distribution function
x = np.array([2.0, 5.0, 10.0], dtype=np.float32)
tx = Tensor(x, dtype=dtype.float32)
cdf = LogNormal.cdf(tx)

# generate samples from the distribution
shape = (3, 2)
sample = LogNormal.sample(shape)

# get information of the distribution
print(LogNormal)
# get information of the underlying distribution and the bijector separately
print("underlying distribution:\n", LogNormal.distribution)
print("bijector:\n", LogNormal.bijector)
# get the computation results
print("cdf:\n", cdf)
print("sample shape:\n", sample.shape)
```

输出为：

```text
TransformedDistribution<
  (_bijector): Exp<exp>
  (_distribution): Normal<mean = 0.0, standard deviation = 1.0>
  >
underlying distribution:
 Normal<mean = 0.0, standard deviation = 1.0>
bijector:
 Exp<exp>
cdf:
 [0.7558914 0.9462397 0.9893489]
sample shape:
(3, 2)
```

当构造 `TransformedDistribution` 映射变换的 `is_constant_jacobian = true` 时（如 `ScalarAffine`)，构造的 `TransformedDistribution` 实例可以使用直接使用 `mean` 接口计算均值，例如：

```python
normal = msd.Normal(0.0, 1.0, dtype=dtype.float32)
scalaraffine = msb.ScalarAffine(1.0, 2.0)
trans_dist = msd.TransformedDistribution(scalaraffine, normal, seed=0)
mean = trans_dist.mean()
print(mean)
```

输出为：

```text
2.0
```

### 图模式下调用TransformedDistribution实例

在图模式下，`TransformedDistribution` 类可用在网络中。

```python
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, dtype
import mindspore.context as context
import mindspore.nn.probability.bijector as msb
import mindspore.nn.probability.distribution as msd
context.set_context(mode=context.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self, shape, dtype=dtype.float32, seed=0, name='transformed_distribution'):
        super(Net, self).__init__()
        # create TransformedDistribution distribution
        self.exp = msb.Exp()
        self.normal = msd.Normal(0.0, 1.0, dtype=dtype)
        self.lognormal = msd.TransformedDistribution(self.exp, self.normal, seed=seed, name=name)
        self.shape = shape

    def construct(self, value):
        cdf = self.lognormal.cdf(value)
        sample = self.lognormal.sample(self.shape)
        return cdf, sample

shape = (2, 3)
net = Net(shape=shape, name="LogNormal")
x = np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32)
tx = Tensor(x, dtype=dtype.float32)
cdf, sample = net(tx)
print("cdf: ", cdf)
print("sample shape: ", sample.shape)
```

输出为：

```text
cdf:  [0.7558914  0.86403143 0.9171715  0.9462397 ]
sample shape:  (2, 3)
```

## 概率分布映射

Bijector（`mindspore.nn.probability.bijector`）是概率编程的基本组成部分。Bijector描述了一种随机变量的变换方法，可以通过一个已有的随机变量X和一个映射函数f生成一个新的随机变量$Y = f(x)$。
`Bijector` 提供了映射相关的四种变换方法。它可以当做算子直接使用，也可以作用在某个随机变量 `Distribution` 类实例上生成新的随机变量的 `Distribution` 类实例。

### Bijector类接口设计

#### Bijector基类

`Bijector` 类是所有概率分布映射的基类。其接口包括：

1. 属性
    - `name`：返回 `name` 的值。
    - `dtype`：返回 `dtype` 的值。
    - `parameters`：返回 `parameter` 的值。
    - `is_constant_jacobian`：返回 `is_constant_jacobian` 的值。
    - `is_injective`：返回 `is_injective` 的值。

2. 映射函数
    - `forward`：正向映射，创建派生类后由派生类的 `_forward` 决定参数。
    - `inverse`：反向映射，创建派生类后由派生类的 `_inverse` 决定参数。
    - `forward_log_jacobian`：正向映射的导数的对数，创建派生类后由派生类的 `_forward_log_jacobian` 决定参数。
    - `inverse_log_jacobian`：反向映射的导数的对数，创建派生类后由派生类的 `_inverse_log_jacobian` 决定参数。

`Bijector` 作为函数调用：输入是一个 `Distribution` 类：生成一个 `TransformedDistribution` **（不可在图内调用）**。

#### 幂函数变换映射(PowerTransform)

`PowerTransform` 做如下变量替换：`Y = g(X) = {(1 + X * power)}^{1 / power}`。其接口包括：

1. 属性
    - `power`：返回 `power` 的值，类型为`Tensor`。

2. 映射函数
    - `forward`：正向映射，输入为 `Tensor` 。
    - `inverse`：反向映射，输入为 `Tensor` 。
    - `forward_log_jacobian`：正向映射的导数的对数，输入为 `Tensor` 。
    - `inverse_log_jacobian`：反向映射的导数的对数，输入为 `Tensor` 。

#### 指数变换映射(Exp)

`Exp` 做如下变量替换：`Y = g(X)= exp(X)`。其接口包括：

映射函数

- `forward`：正向映射，输入为 `Tensor` 。
- `inverse`：反向映射，输入为 `Tensor` 。
- `forward_log_jacobian`：正向映射的导数的对数，输入为 `Tensor` 。
- `inverse_log_jacobian`：反向映射的导数的对数，输入为 `Tensor` 。

#### 标量仿射变换映射(ScalarAffine)

`ScalarAffine` 做如下变量替换：`Y = g(X) = scale * X + shift`。其接口包括：

1. 属性
    - `scale`：返回`scale`的值，类型为`Tensor`。
    - `shift`：返回`shift`的值，类型为`Tensor`。

2. 映射函数
    - `forward`：正向映射，输入为 `Tensor` 。
    - `inverse`：反向映射，输入为 `Tensor` 。
    - `forward_log_jacobian`：正向映射的导数的对数，输入为 `Tensor` 。
    - `inverse_log_jacobian`：反向映射的导数的对数，输入为 `Tensor` 。

#### Softplus变换映射(Softplus)

`Softplus` 做如下变量替换：`Y = g(X) = log(1 + e ^ {sharpness * X}) / sharpness`。其接口包括：

1. 属性
    - `sharpness`：返回 `sharpness` 的值，类型为`Tensor`。

2. 映射函数
    - `forward`：正向映射，输入为 `Tensor` 。
    - `inverse`：反向映射，输入为 `Tensor` 。
    - `forward_log_jacobian`：正向映射的导数的对数，输入为 `Tensor` 。
    - `inverse_log_jacobian`：反向映射的导数的对数，输入为 `Tensor` 。

#### 耿贝尔累计密度函数映射(GumbelCDF)

`GumbelCDF` 做如下变量替换：$Y = g(X) = \exp(-\exp(-\frac{X - loc}{scale}))$。其接口包括：

1. 属性
    - `loc`：返回`loc`的值，类型为`Tensor`。
    - `scale`：返回`scale`的值，类型为`Tensor`。

2. 映射函数
    - `forward`：正向映射，输入为 `Tensor` 。
    - `inverse`：反向映射，输入为 `Tensor` 。
    - `forward_log_jacobian`：正向映射的导数的对数，输入为 `Tensor` 。
    - `inverse_log_jacobian`：反向映射的导数的对数，输入为 `Tensor` 。

#### 逆映射(Invert)

`Invert` 对一个映射做逆变换，其接口包括：

1. 属性
    - `bijector`：返回初始化时使用的*Bijector*，类型为`Bijector`。

2. 映射函数
    - `forward`：正向映射，输入为 `Tensor` 。
    - `inverse`：反向映射，输入为 `Tensor` 。
    - `forward_log_jacobian`：正向映射的导数的对数，输入为 `Tensor` 。
    - `inverse_log_jacobian`：反向映射的导数的对数，输入为 `Tensor` 。

### PyNative模式下调用Bijector实例

在执行之前，我们需要导入需要的库文件包。双射类最主要的库是 `mindspore.nn.probability.bijector`，导入后我们使用 `msb` 作为库的缩写并进行调用。

下面我们以 `PowerTransform` 为例。创建一个指数为2的 `PowerTransform` 对象。

```python
import numpy as np
import mindspore.nn as nn
import mindspore.nn.probability.bijector as msb
import mindspore.context as context
from mindspore import Tensor, dtype

context.set_context(mode=context.PYNATIVE_MODE)

powertransform = msb.PowerTransform(power=2.)

x = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
tx = Tensor(x, dtype=dtype.float32)
forward = powertransform.forward(tx)
inverse = powertransform.inverse(tx)
forward_log_jaco = powertransform.forward_log_jacobian(tx)
inverse_log_jaco = powertransform.inverse_log_jacobian(tx)

print(powertransform)
print("forward: ", forward)
print("inverse: ", inverse)
print("forward_log_jacobian: ", forward_log_jaco)
print("inverse_log_jacobian: ", inverse_log_jaco)
```

输出：

```text
PowerTransform<power = 2.0>
forward:  [2.236068  2.6457515 3.        3.3166249]
inverse:  [ 1.5       4.        7.5      12.000001]
forward_log_jacobian:  [-0.804719  -0.9729551 -1.0986123 -1.1989477]
inverse_log_jacobian:  [0.6931472 1.0986123 1.3862944 1.609438 ]
```

### 图模式下调用Bijector实例

在图模式下，`Bijector` 子类可用在网络中。

```python
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.context as context
import mindspore.nn.probability.bijector as msb
context.set_context(mode=context.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        # create a PowerTransform bijector
        self.powertransform = msb.PowerTransform(power=2.)

    def construct(self, value):
        forward = self.powertransform.forward(value)
        inverse = self.powertransform.inverse(value)
        forward_log_jaco = self.powertransform.forward_log_jacobian(value)
        inverse_log_jaco = self.powertransform.inverse_log_jacobian(value)
        return forward, inverse, forward_log_jaco, inverse_log_jaco

net = Net()
x = np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32)
tx = Tensor(x, dtype=mstype.float32)
forward, inverse, forward_log_jaco, inverse_log_jaco = net(tx)
print("forward: ", forward)
print("inverse: ", inverse)
print("forward_log_jacobian: ", forward_log_jaco)
print("inverse_log_jacobian: ", inverse_log_jaco)
```

输出为：

```text
forward:  [2.236068  2.6457515 3.        3.3166249]
inverse:  [ 1.5       4.        7.5      12.000001]
forward_log_jacobian:  [-0.804719  -0.9729551 -1.0986123 -1.1989477]
inverse_log_jacobian:  [0.6931472 1.0986123 1.3862944 1.609438 ]
```

## 深度概率网络

使用MindSpore深度概率编程库（`mindspore.nn.probability.dpn`）来构造变分自编码器（VAE）进行推理尤为简单。我们只需要自定义编码器和解码器（DNN模型），调用VAE或CVAE接口形成其派生网络，然后调用ELBO接口进行优化，最后使用SVI接口进行变分推理。这样做的好处是，不熟悉变分推理的用户可以像构建DNN模型一样来构建概率模型，而熟悉的用户可以调用这些接口来构建更为复杂的概率模型。VAE的接口在`mindspore.nn.probability.dpn`下面，dpn代表的是Deep probabilistic network，这里提供了一些基本的深度概率网络的接口，例如VAE。

### VAE

首先，我们需要先自定义encoder和decoder，调用`mindspore.nn.probability.dpn.VAE`接口来构建VAE网络，我们除了传入encoder和decoder之外，还需要传入encoder输出变量的维度hidden size，以及VAE网络存储潜在变量的维度latent size，一般latent size会小于hidden size。

```python
import mindspore.nn as nn
import mindspore.ops as ops
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
        self.reshape = ops.Reshape()

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

```python
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn.probability.dpn import ConditionalVAE

IMAGE_SHAPE = (-1, 1, 32, 32)


class Encoder(nn.Cell):
    def __init__(self, num_classes):
        super(Encoder, self).__init__()
        self.fc1 = nn.Dense(1024 + num_classes, 400)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.concat = ops.Concat(axis=1)
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
        self.reshape = ops.Reshape()

    def construct(self, z):
        z = self.fc1(z)
        z = self.reshape(z, IMAGE_SHAPE)
        z = self.sigmoid(z)
        return z


encoder = Encoder(num_classes=10)
decoder = Decoder()
cvae = ConditionalVAE(encoder, decoder, hidden_size=400, latent_size=20, num_classes=10)
```

加载数据集，我们可以使用Mnist数据集，具体的数据加载和预处理过程可以参考这里[实现一个图片分类应用](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/quick_start/quick_start.html)，这里会用到create_dataset函数创建数据迭代器。

```python
ds_train = create_dataset(image_path, 128, 1)
```

接下来，需要用到infer接口进行VAE网络的变分推断。

## 概率推断算法

调用ELBO接口（`mindspore.nn.probability.infer.ELBO`）来定义VAE网络的损失函数，调用`WithLossCell`封装VAE网络和损失函数，并定义优化器，之后传入SVI接口（`mindspore.nn.probability.infer.SVI`）。SVI的`run`函数可理解为VAE网络的训练，可以指定训练的`epochs`，返回结果为训练好的网络；`get_train_loss`函数可以返回训练好后模型的loss。

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

```text
The shape of the generated sample is  (64, 1, 32, 32)
```

ConditionalVAE训练过程和VAE的过程类似，但需要注意的是使用训练好的ConditionalVAE网络生成新样本和重建新样本时，需要输入标签信息，例如下面生成的新样本就是64个0-7的数字。

```python
sample_label = Tensor([i for i in range(0, 8)] * 8, dtype=mstype.int32)
generated_sample = cvae.generate_sample(sample_label, 64, IMAGE_SHAPE)
for sample in ds_train.create_dict_iterator():
    sample_x = Tensor(sample['image'], dtype=mstype.float32)
    sample_y = Tensor(sample['label'], dtype=mstype.int32)
    reconstructed_sample = cvae.reconstruct_sample(sample_x, sample_y)
print('The shape of the generated sample is ', generated_sample.shape)
```

查看一下新生成的样本的shape：

```text
The shape of the generated sample is  (64, 1, 32, 32)
```

如果希望新生成的样本更好，更清晰，用户可以自己定义更复杂的encoder和decoder，这里的示例只用了两层全连接层，仅供示例的指导。

## 贝叶斯层

下面的范例使用MindSpore的`nn.probability.bnn_layers`中的API实现BNN图片分类模型。MindSpore的`nn.probability.bnn_layers`中的API包括`NormalPrior`，`NormalPosterior`，`ConvReparam`，`DenseReparam`，`DenseLocalReparam`和`WithBNNLossCell`。BNN与DNN的最大区别在于，BNN层的weight和bias不再是确定的值，而是服从一个分布。其中，`NormalPrior`，`NormalPosterior`分别用来生成服从正态分布的先验分布和后验分布；`ConvReparam`和`DenseReparam`分别是使用reparameterization方法实现的贝叶斯卷积层和全连接层；`DenseLocalReparam`是使用Local Reparameterization方法实现的贝叶斯全连接层；`WithBNNLossCell`是用来封装BNN和损失函数的。

如何使用`nn.probability.bnn_layers`中的API构建贝叶斯神经网络并实现图片分类，可以参考教程[使用贝叶斯网络](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/apply_deep_probability_programming.html#id3)。

## 贝叶斯转换

对于不熟悉贝叶斯模型的研究人员，MDP提供了贝叶斯转换接口（`mindspore.nn.probability.transform`），支持DNN (Deep Neural Network)模型一键转换成BNN (Bayesian Neural Network)模型。

其中的模型转换API`TransformToBNN`的`__init__`函数定义如下：

```python
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
API`TransformToBNN`主要实现了两个功能：

- 功能一：转换整个模型

  `transform_to_bnn_model`方法可以将整个DNN模型转换为BNN模型。其定义如下：

  ```python
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

  参数`get_dense_args`指定从DNN模型的全连接层中获取哪些参数，默认值是DNN模型的全连接层和BNN的全连接层所共有的参数，参数具体的含义可以参考[API说明文档](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/nn/mindspore.nn.Dense.html)；`get_conv_args`指定从DNN模型的卷积层中获取哪些参数，默认值是DNN模型的卷积层和BNN的卷积层所共有的参数，参数具体的含义可以参考[API说明文档](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/nn/mindspore.nn.Conv2d.html)；参数`add_dense_args`和`add_conv_args`分别指定了要为BNN层指定哪些新的参数值。需要注意的是，`add_dense_args`中的参数不能与`get_dense_args`重复，`add_conv_args`和`get_conv_args`也是如此。

- 功能二：转换指定类型的层

  `transform_to_bnn_layer`方法可以将DNN模型中指定类型的层（`nn.Dense`或者`nn.Conv2d`）转换为对应的贝叶斯层。其定义如下：

  ```python
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

如何在MindSpore中使用API`TransformToBNN`可以参考教程[DNN一键转换成BNN](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/apply_deep_probability_programming.html#dnnbnn)

## 贝叶斯工具箱

### 不确定性评估

贝叶斯神经网络的优势之一就是可以获取不确定性，MDP在上层提供了不确定性估计的工具箱（`mindspore.nn.probability.toolbox`），用户可以很方便地使用该工具箱计算不确定性。不确定性意味着深度学习模型对预测结果的不确定程度。目前，大多数深度学习算法只能给出高置信度的预测结果，而不能判断预测结果的确定性，不确定性主要有两种类型：偶然不确定性和认知不确定性。

- 偶然不确定性（Aleatoric Uncertainty）：描述数据中的内在噪声，即无法避免的误差，这个现象不能通过增加采样数据来削弱。
- 认知不确定性（Epistemic Uncertainty）：模型自身对输入数据的估计可能因为训练不佳、训练数据不够等原因而不准确，可以通过增加训练数据等方式来缓解。

不确定性评估工具箱的接口如下：

- `model`：待评估不确定性的已训练好的模型。
- `train_dataset`：用于训练的数据集，迭代器类型。
- `task_type`：模型的类型，字符串，输入“regression”或者“classification”。
- `num_classes`：如果是分类模型，需要指定类别的标签数量。
- `epochs`：用于训练不确定模型的迭代数。
- `epi_uncer_model_path`：用于存储或加载计算认知不确定性的模型的路径。
- `ale_uncer_model_path`：用于存储或加载计算偶然不确定性的模型的路径。
- `save_model`：布尔类型，是否需要存储模型。

在使用前，需要先训练好模型，以LeNet5为例，使用方式如下：

```python
from mindspore.nn.probability.toolbox.uncertainty_evaluation import UncertaintyEvaluation
from mindspore import load_checkpoint, load_param_into_net

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
    print('The shape of aleatoric uncertainty is ', aleatoric_uncertainty.shape)
```

`eval_epistemic_uncertainty`计算的是认知不确定性，也叫模型不确定性，对于每一个样本的每个预测标签都会有一个不确定值；`eval_aleatoric_uncertainty`计算的是偶然不确定性，也叫数据不确定性，对于每一个样本都会有一个不确定值。
所以输出为：

```text
The shape of epistemic uncertainty is (32, 10)
The shape of aleatoric uncertainty is (32,)
```

uncertainty的值大于等于0，越大表示不确定性越高。

### 异常检测

异常检测(Anomaly Detection)可以找到与“主要数据分布不同”的异常值，比如在数据预处理中找出异常点，有助于提升模型的拟合能力。

MDP在上层基于变分自编码器（VAE）提供了异常检测的工具箱(`VAEAnomalyDetection`)，与VAE的使用类似，我们只需要自定义编码器和解码器（DNN模型），初始化相关参数，便可以使用该工具箱检测异常点。

基于VAE的异常检测工具箱的接口如下：

- `encoder`：编码器（Cell类型）
- `decoder`：解码器（Cell类型）
- `hidden_size`：编码器输出张量的大小
- `latent_size`：隐空间的大小

编码器和解码器可使用以上的Encoder和Decoder，设置hidden_size和latent_size，进行类的初始化，之后传入数据集可以进行异常点的检测。

```python
from mindspore.nn.probability.toolbox.vae_anomaly_detection import VAEAnomalyDetection

if __name__ == '__main__':
    encoder = Encoder()
    decoder = Decoder()
    ood = VAEAnomalyDetection(encoder=encoder, decoder=decoder,
                              hidden_size=400, latent_size=20)
    ds_train = create_dataset('workspace/mnist/train')
    ds_eval = create_dataset('workspace/mnist/test')
    model = ood.train(ds_train)
    for sample in ds_eval.create_dict_iterator(output_numpy=True, num_epochs=1):
        sample_x = Tensor(sample['image'], dtype=mstype.float32)
        score = ood.predict_outlier_score(sample_x)
        outlier = ood.predict_outlier(sample_x)
        print(score, outlier)
```

`score`输出的是样本的异常分数；`outlier`是布尔类型，True代表是异常点，False代表不是异常点。
