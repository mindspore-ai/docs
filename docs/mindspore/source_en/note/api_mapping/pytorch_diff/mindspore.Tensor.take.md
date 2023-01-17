# Function Differences with torch.take

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/mindspore.Tensor.take.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.take

```python
torch.take(input, index)
```

For more information, see [torch.take](https://pytorch.org/docs/1.5.0/torch.html#torch.take).

## mindspore.Tensor.take

```python
mindspore.Tensor.take(indices, axis=None, mode="clip")
```

For more information, see [mindspore.Tensor.take](https://mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.take.html#mindspore.Tensor.take).

## Uasge

The basic function is to get the corresponding element from the input Tensor based on the index passed in.

`torch.take` first stretches the original Tensor, and then gets the elements according to `index`, which is set to be smaller than the number of elements in the input Tensor.

The default state of `mindspore.Tensor.take` (`axis=None`) also does a `ravel` operation on the Tensor first, and then returns the elements according to `indices`. In addition, you can set `axis` to select elements according to the specified `axis`. The value of `indices` can exceed the number of Tensor elements, so you can set a different return strategy by input parameter `mode`. Please refer to the API notes for details.

## Code Example

```python
import mindspore as ms
import numpy as np

a = ms.Tensor([[1, 2, 8],[3, 4, 6]], ms.float32)
indices = ms.Tensor(np.array([1, 10]))
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
