# 比较与torch.nn.functional.leaky_relu的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/leaky_relu.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.functional.leaky_relu

```text
torch.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor
```

更多内容详见[torch.nn.functional.leaky_relu](https://pytorch.org/docs/1.8.1/nn.functional.html#leaky-relu)。

## mindspore.ops.leaky_relu

```text
mindspore.ops.leaky_relu(x, alpha=0.2) -> Tensor
```

更多内容详见[mindspore.ops.leaky_relu](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.leaky_relu.html#mindspore.ops.leaky_relu)。

## 差异对比

PyTorch：leaky_relu激活函数。`input` 中小于0的元素乘以 `negative_slope` 。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。不同的是，MindSpore中 `alpha` 的初始值是0.2，PyTorch中对应的 `negative_slope` 初始值是0.01。

| 分类 | 子类  | PyTorch      | MindSpore    | 差异                          |
| ---- | ----- | ------------ | ------------ | ---------------------------- |
| 参数 | 参数1 | input | x  | 功能一致，参数名不同          |
|      | 参数2 | negative_slope | alpha | 功能一致，参数名不同，默认值不同 |
|      | 参数3 | inplace | -     | 是否对入参进行原地修改，MindSpore无此功能 |

### 代码示例

```python
# PyTorch
import torch

input = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32)
output = torch.nn.functional.leaky_relu(input, negative_slope=0.5, inplace=False)
print(output)
# tensor([-1.0000, -0.5000,  0.0000,  1.0000,  2.0000])

# MindSpore
import mindspore

input = mindspore.Tensor([-2, -1, 0, 1, 2], dtype=mindspore.float32)
output = mindspore.ops.leaky_relu(input, alpha=0.5)
print(output)
# [-1.  -0.5  0.   1.   2. ]
```
