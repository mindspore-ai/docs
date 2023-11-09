# 比较与torch.Tensor.repeat的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/tensor_repeat.md)

## torch.Tensor.repeat

```python
torch.Tensor.repeat(*sizes)
```

更多内容详见[torch.Tensor.repeat](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.repeat)。

## mindspore.Tensor.tile

```python
mindspore.Tensor.tile(multiples)
```

更多内容详见[mindspore.Tensor.tile](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/mindspore/Tensor/mindspore.Tensor.tile.html)。

## 差异对比

接口 `mindspore.Tensor.tile` 的使用方式和 `torch.Tensor.repeat` 基本一致。

| 分类       | 子类         | PyTorch      | MindSpore      | 差异          |
| ---------- | ------------ | ------------ | ---------      | ------------- |
| 参数       | 参数 1       | *sizes         | multiples          | PyTorch的参数类型是 torch.Size 或 int；MindSpore的参数类型必须是 tuple。 |

### 代码示例

```python
# PyTorch
import torch

input = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
output = input.repeat(16, 1)
print(output.shape)
# torch.Size([32, 2])

# MindSpore
import mindspore

x = mindspore.Tensor([[1, 2], [3, 4]], dtype=mindspore.float32)
output = x.tile((16, 1))
print(output.shape)
# (32, 2)
```
