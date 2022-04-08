# 比较与torch.take的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/TensorTake.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.take

```python
torch.take(input, index)
```

更多内容详见[torch.take](https://pytorch.org/docs/1.5.0/torch.html#torch.take)。

## mindspore.Tensor.take

```python
mindspore.Tensor.take(indices, axis=None, mode="clip")
```

更多内容详见[mindspore.Tensor.take](https://mindspore.cn/docs/api/zh-CN/master/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.take)。

## 使用方式

基础功能为根据传入的索引从输入Tensor中获取对应的元素。

`torch.take`首先将原始Tensor拉长，然后根据`index`获取元素，`index`设置值需小于输入Tensor的元素数。

`mindspore.Tensor.take`默认状态下（`axis=None`）同样先对Tensor做`ravel`操作，再按照`indices`返回元素。除此之外，可以通过`axis`设定按照指定`axis`选取元素。`indices`数值可以超出Tensor元素数目，此时可以通过入参`mode`设置不同的返回策略，具体说明请参考API注释。

## 代码示例

```python
import mindspore
from mindspore import Tensor
import numpy as np

a = Tensor([[1, 2, 8],[3, 4, 6]], mindspore.float32)
indices = Tensor(np.array([1, 10]))
# take(self, indices, axis=None, mode='clip'):
print(a.take(indices))
# [2. 6.]
print(a.take(indices, axis=1))
# [[2. 8.]
#  [4. 6.]]
print(a.take(indices, mode="wrap"))
# [2. 4.]

import torch
b = torch.tensor([[1, 2, 8],[3, 4, 6]])
indices = torch.tensor([1, 5])
print(torch.take(b, indices))
# tensor([2, 6])
```
