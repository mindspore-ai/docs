# 比较与torch.Tensor.sum的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/TensorSum.md)

## torch.Tensor.sum

```python
torch.Tensor.sum(dim=None, keepdim=False, dtype=None)
```

更多内容详见[torch.Tensor.sum](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.sum)。

## mindspore.Tensor.sum

```python
mindspore.Tensor.sum(axis=None, dtype=None, keepdims=False, initial=None)
```

更多内容详见[mindspore.Tensor.sum](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/Tensor/mindspore.Tensor.sum.html#mindspore.Tensor.sum)。

## 差异对比

MindSpore此API功能与PyTorch一致，参数的个数和顺序不一致。

PyTorch：无参数 `initial` 。参数 `keepdim` 和 `dtype` 的相对顺序与MindSpore不同。

MindSpore：可以通过参数 `initial` 配置求和的起始值。参数 `keepdim` 和 `dtype` 的相对顺序与PyTorch不同。

| 分类 | 子类  | PyTorch | MindSpore | 差异                    |
| ---- | ----- |---------|-----------| ----------------------- |
| 参数 | 参数1 | dim | axis | 二者参数名不同，均表示求和的指定维度 |
|      | 参数2 | keepdim | dtype | 二者参数 `keepdim` 和 `dtype` 的相对顺序不同 |
|      | 参数3 | dtype | keepdims | 二者参数 `keepdims` 和 `dtype` 的相对顺序不同 |
|      | 参数4 | - | initial | MindSpore可以通过参数 `initial` 配置求和的起始值，PyTorch无参数 `initial` |

## 代码示例

```python
# PyTorch
import torch

b = torch.Tensor([10, -5])
print(torch.Tensor.sum(b))
# tensor(5.)

# MindSpore
import mindspore as ms

a = ms.Tensor([10, -5], ms.float32)
print(a.sum())
# 5.0
print(a.sum(initial=2))
# 7.0
```