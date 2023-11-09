# 比较与torch.nn.functional.gelu的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/mindspore.ops.gelu.md)

## torch.nn.functional.gelu

```text
torch.nn.functional.gelu(input) -> Tensor
```

更多内容详见[torch.nn.functional.gelu](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.gelu)。

## mindspore.ops.gelu

```text
mindspore.ops.gelu(input_x, approximate='none')
```

更多内容详见[mindspore.ops.gelu](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/ops/mindspore.ops.gelu.html)。

## 差异对比

PyTorch：该函数表示高斯误差线性单位函数$GELU(X)=X\times \Phi(x)$，其中$\Phi(x)$是高斯分布的积累分布函数。输入x表示任意数量的维度。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。

| 分类 | 子类  | PyTorch | MindSpore   | 差异                                                         |
| ---- | ----- | ------- | ----------- | ------------------------------------------------------------ |
|   参数   | 参数1 |    -     | approximate | gelu近似算法。有两种：'none' 和 'tanh'。默认值：none。经测试，approximate为 'none' 后，输出结果与Pytorch相似。 |
| 输入 | 单输入 | input      | input_x           | 功能一致，参数名不同               |

### 代码示例1

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
input = torch.Tensor([[2, 4], [1, 2]])
output = torch.nn.functional.gelu(input)
print(output.detach().numpy())
# [[1.9544997 3.9998734]
#  [0.8413447 1.9544997]]

# MindSpore
import mindspore
import numpy as np
x = mindspore.Tensor(np.array([[2, 4], [1, 2]]), mindspore.float32)
output = mindspore.ops.gelu(x)
print(output)
# [[1.9545997 3.99993]
#  [0.841192 1.9545977]]
```
