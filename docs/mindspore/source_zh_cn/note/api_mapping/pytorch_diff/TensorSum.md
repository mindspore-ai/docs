# 比较与torch.Tensor.sum的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/TensorSum.md)

## torch.Tensor.sum

```python
torch.Tensor.sum(dim=None, keepdim=False, dtype=None)
```

更多内容详见[torch.Tensor.sum](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.sum)。

## mindspore.Tensor.sum

```python
mindspore.Tensor.sum(axis=None, dtype=None, keepdims=False, initial=None)
```

更多内容详见[mindspore.Tensor.sum](https://www.mindspore.cn/docs/en/r1.7/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.sum)。

## 使用方式

基本功能一致。`mindspore.Tensor.sum`可以通过入参`initial`配置求和的起始值，其他入参两接口设定相同。

## 代码示例

```python
from mindspore import Tensor
import mindspore

a = Tensor([10, -5], mindspore.float32)
print(a.sum()) # 5.0
print(a.sum(initial=2)) # 7.0

import torch
b = torch.Tensor([10, -5])
print(torch.Tensor.sum(b)) # tensor(5.)
```
