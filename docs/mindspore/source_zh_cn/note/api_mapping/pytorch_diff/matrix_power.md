# 比较与torch.matrix_power的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/matrix_power.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

> `mindspore.Tensor.matrix_power` 和 `torch.Tensor.matrix_power` 的功能差异，参考 `mindspore.ops.matrix_power` 和 `torch.matrix_power` 的功能差异比较。

## torch.matrix_power

```python
torch.matrix_power(input, n)
```

更多内容详见[torch.matrix_power](https://pytorch.org/docs/1.8.1/generated/torch.matrix_power.html)。

## mindspore.ops.matrix_power

```python
mindspore.ops.matrix_power(x, n)
```

更多内容详见[mindspore.ops.matrix_power](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.matrix_power.html)。

## 差异对比

PyTorch：

- 输入tensor的维度可以是2维或更高。

- 输入tensor支持的数据类型为uint8、int8/16/32/64和float16/32/64。

MindSpore：

- 输入tensor的维度只能是3维。

- 输入tensor支持的数据类型为float16和float32。

功能上无差异。

| 分类       | 子类         | PyTorch      | MindSpore  | 差异          |
| ---------- | ------------ | ------------ | ---------  | ------------- |
| 参数       | 参数 1       | input         | x        | MindSpore仅支持3维，float16和float32类型，PyTorch支持2维或更高维度，uint8、int8/16/32/64和float16/32/64类型。 |
|            | 参数 2       | n             | n        | 功能一致        |

## 差异分析与示例

```python
# PyTorch
import torch
x = torch.tensor([[0, 1], [-1, 0]], dtype=torch.int32)
y = torch.matrix_power(x, 2)
print(x.shape)
print(y)
# torch.Size([2, 2])
# tensor([[-1,  0],
#         [ 0, -1]], dtype=torch.int32)

# MindSpore
import mindspore as ms
x = ms.Tensor([[[0, 1], [-1, 0]]], dtype=ms.float32)
y = ms.ops.matrix_power(x, 2)
print(x.shape)
print(y)
# (1, 2, 2)
# [[[-1.  0.]
#   [-0. -1.]]]
```
