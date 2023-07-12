# 比较与torch.bucketize的差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/bucketize.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.bucketize

```python
torch.bucketize(input, boundaries, *, out_int32=False, right=False, out=None)
```

更多内容详见[torch.bucketize](https://pytorch.org/docs/1.8.1/torch.html#torch.bucketize)。

## mindspore.ops.bucketize

```python
class mindspore.ops.bucketize(input, boundaries, *, right=False)
```

更多内容详见[mindspore.ops.bucketize](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.bucketize.html#mindspore.ops.bucketize)。

## 使用方式

MindSpore此API功能与PyTorch一致，参数支持的数据类型有差异。

PyTorch：`input` 支持scalar和Tensor类型，`boundaries` 支持Tensor类型，且可以通过 `out_int32` 指定返回的索引的数据类型。

MindSpore：`input` 支持Tensor类型，`boundaries` 支持list类型，无 `out_int32` 参数。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                    |
| ---- | ----- | ------- | --------- | --------------------------------------- |
| 参数 | 参数1 | input   | input         | 功能一致，支持数据类型不同                    |
|      | 参数2 | boundaries   | boundaries      | 功能一致，支持数据类型不同 |
|      | 参数3 | out_int32   | - | PyTorch的 `out_int32` 可以指定返回的索引类型，MindSpore无此参数 |
|      | 参数4 | right   | right | 一致 |
|      | 参数5 | out   | -         | PyTorch的 `out` 可以获取输出，MindSpore无此参数 |

## 代码示例

```python
import torch

boundaries = torch.tensor([1, 3, 5, 7, 9])
v = torch.tensor([[3, 6, 9], [3, 6, 9]])
out1 = torch.bucketize(v, boundaries)
out2 = torch.bucketize(v, boundaries, right=True)
print(out1)
# out:
# tensor([[1, 3, 4],
#        [1, 3, 4]])

print(out2)
# out:
# tensor([[2, 3, 5],
#        [2, 3, 5]])

from mindspore import Tensor, ops
boundaries = [1, 3, 5, 7, 9]
v = Tensor([[3, 6, 9], [3, 6, 9]])
out1 = ops.bucketize(v, boundaries)
out2 = ops.bucketize(v, boundaries, right=True)
print(out1)
# out:
# [[1 3 4]
#  [1 3 4]]

print(out2)
# out:
# [[2 3 5]
#  [2 3 5]]
```