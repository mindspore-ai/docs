# 比较与torch.nn.functional.normalize的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/L2Normalize.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source.png"></a>

## torch.nn.functional.normalize

```python
torch.nn.functional.normalize(
    input,
    p=2,
    dim=1,
    eps=1e-12,
    out=None
)
```

更多内容详见[torch.nn.functional.normalize](https://pytorch.org/docs/1.5.0/nn.functional.html#torch.nn.functional.normalize)。

## mindspore.ops.L2Normalize

```python
class mindspore.ops.L2Normalize(
    axis=0,
    epsilon=1e-4
)(input_x)
```

更多内容详见[mindspore.ops.L2Normalize](https://mindspore.cn/docs/zh-CN/r1.11/api_python/ops/mindspore.ops.L2Normalize.html#mindspore.ops.L2Normalize)。

## 差异对比

PyTorch：支持通过指定参数`p`来使用Lp范式。计算公式为输入作为分子，输入的平方和先开方后再求对应epsilon的max作为分母。函数定义如下：

$$
v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.
$$

MindSpore：仅支持L2范式。计算公式为输入作为分子，输入的平方和先求对应epsilon的max后再开方作为分母。函数定义如下：

$$
\displaylines{{\text{output} = \frac{x}{\sqrt{\text{max}( \sum_{i}^{}\left | x_i  \right | ^2, \epsilon)}}}}
$$

因为计算公式差异导致在某些输入下的计算结果存在差异。

### 代码示例1

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, you can directly pass data into the function, and the default dimension is 0.
l2_normalize = ops.L2Normalize()
input_x = ms.Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
output = l2_normalize(input_x)
print(output)
# [0.2673 0.5345 0.8018]

# In torch, parameter p should be set to determine it is a lp normalization, and the default dimension is 1.
input_x = torch.tensor(np.array([1.0, 2.0, 3.0]))
outputL2 = torch.nn.functional.normalize(input=input_x, p=2, dim=0)
outputL3 = torch.nn.functional.normalize(input=input_x, p=3, dim=0)
print(outputL2)
# tensor([0.2673, 0.5345, 0.8018], dtype=torch.float64)
print(outputL3)
# tensor([0.3029, 0.6057, 0.9086], dtype=torch.float64)
```

### 代码示例2

> 两API计算公式存在差异，当输入数据特别小的时候，会产生巨大的结果差异。

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, you can directly pass data into the function, and the default dimension is 0.
l2_normalize = ops.L2Normalize()
input_x = ms.Tensor(np.array([1.0 * 1e-10, 2.0 * 1e-10, 3.0 * 1e-10]), ms.float32)
output = l2_normalize(input_x)
print(output)
# [1.e-08 2.e-08 3.e-08]

# In torch, parameter p should be set to determine it is a lp normalization, and the default dimension is 1.
input_x = torch.tensor(np.array([1.0 * 1e-10, 2.0 * 1e-10, 3.0 * 1e-10]))
outputL2 = torch.nn.functional.normalize(input=input_x, p=2, dim=0)
outputL3 = torch.nn.functional.normalize(input=input_x, p=3, dim=0)
print(outputL2)
# tensor([0.2673, 0.5345, 0.8018], dtype=torch.float64)
print(outputL3)
# tensor([0.3029, 0.6057, 0.9086], dtype=torch.float64)
```