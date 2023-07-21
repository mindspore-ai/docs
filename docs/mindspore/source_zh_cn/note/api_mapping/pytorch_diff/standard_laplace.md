# 比较与torch.distributions.laplace.Laplace的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/standard_laplace.md)

## torch.distributions.laplace.Laplace

```text
torch.distributions.laplace.Laplace(loc, scale) -> Class Instance
```

更多内容详见[torch.distributions.laplace.Laplace](https://pytorch.org/docs/1.8.1/distributions.html#torch.distributions.laplace.Laplace)。

## mindspore.ops.standard_laplace

```text
mindspore.ops.standard_laplace(shape, seed=None) -> Tensor
```

更多内容详见[mindspore.ops.standard_laplace](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.standard_laplace.html)。

## 差异对比

PyTorch：创建一个Laplace分布实例，调用该实例sample接口进行采样可以生成符合Laplace分布的随机值。

MindSpore：生成符合标准Laplace（mean=0, lambda=1）分布的随机数。当PyTorch中loc=0，scale=1，sample函数输入shape与MindSpore一致时，两API实现功能一致。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                    |
| ---- | ----- | ------- | --------- | --------------------------------------- |
| 参数 | 参数1 | loc   | -         | MindSpore无此参数，默认实现loc=0的功能                    |
|      | 参数2 | scale   | -      | MindSpore无此参数，默认实现scale=1的功能 |
|      | 参数3 | -   | shape | PyTorch这个参数在调用sample接口时传入 |
|      | 参数4 | -   | seed        | 算子层的随机种子，PyTorch无此参数 |

### 代码示例

> PyTorch中每一个生成的随机值占用一维度，因此在MindSpore中传入的shape最内层增加一个长度为1的维度，两API实现功能一致。

```python
# PyTorch
import torch

m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
shape = (4, 4)
sample = m.sample(shape)
print(tuple(sample.shape))
# (4, 4, 1)

# MindSpore
import mindspore
from mindspore import ops

shape = (4, 4, 1)
output = ops.standard_laplace(shape)
result = output.shape
print(result)
# (4, 4, 1)
```
