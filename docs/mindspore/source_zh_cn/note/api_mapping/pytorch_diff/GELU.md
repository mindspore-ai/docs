# 比较与torch.nn.GELU的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/GELU.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.GELU

```python
class torch.nn.GELU()(input) -> Tensor
```

更多内容详见[torch.nn.GELU](https://pytorch.org/docs/1.8.1/generated/torch.nn.GELU.html)。

## mindspore.nn.GELU

```python
class mindspore.nn.GELU(approximate=True)(x) -> Tensor
```

更多内容详见[mindspore.nn.GELU](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.GELU.html)。

## 差异对比

PyTorch：该函数表示高斯误差线性单位函数$GELU(X)=X\times \Phi(x)$，其中$\Phi(x)$是高斯分布的积累分布函数。输入x表示任意数量的维度。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。

| 分类 | 子类  | PyTorch | MindSpore   | 差异                                                         |
| ---- | ----- | ------- | ----------- | ------------------------------------------------------------ |
| 参数 | 参数1 | input      | x           | 接口输入，功能一致，仅参数名不同               |
|      | 参数2 |    -     | approximate | 决定是否启用approximation，默认值为True，如果approximate的值为True，高斯误差线性激活函数为：$0.5\times x\times (1+tanh(aqrt(2/pi)\times (x+0.044715 \times x^{3})))$，否则为：$x \times P(X\leqslant x)=0.5\times x \times (1+erf(x/sqrt(2)))$，其中$P(X)\sim N(0,1)$ |

### 代码示例1

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
input = torch.Tensor([[2, 4], [1, 2]])
output = torch.nn.GELU()(input)
print(output.detach().numpy())
# [[1.9544997 3.9998734]
#  [0.8413447 1.9544997]]

# MindSpore
import mindspore
import numpy as np
x = mindspore.Tensor(np.array([[2, 4], [1, 2]]), mindspore.float32)
output = mindspore.nn.GELU()(x)
print(output)
# [[1.9545977 3.99993 ]
#  [0.841192 1.9545977]]
```
