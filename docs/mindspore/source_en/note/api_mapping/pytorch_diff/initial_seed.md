# Differences with torch.initial_seed

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/initial_seed.md)

## torch.initial_seed

```text
class torch.initial_seed
```

For more information, see [torch.initial_seed](https://pytorch.org/docs/1.8.1/generated/torch.initial_seed.html).

## mindspore.nn.initial_seed

```text
class mindspore.nn.initial_seed
```

For more information, see [mindspore.nn.initial_seed](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.initial_seed.html).

## Differences

PyTorch: `initial_seed` returns the initialization seed for the default generator. The generator state contains the values `seed` and `offset`, where `seed` provides the seed for the random operator to generate a random sequence and `offset` provides the offset on the random sequence. When `seed` and `offset` are fixed, the randomizer generates fixed values.

MindSpore: The implementation function of API in MindSpore is basically the same as that of PyTorch. MindSpore returns the Tensor type and PyTorch returns the int type.

### Code Example 1

> The two APIs implement the same function and have the same usage.

```python
#PyTorch
import torch

torch.manual_seed(12)
torch_seed = torch.initial_seed()
print(torch_seed)
# 12


# MindSpore
from mindspore.nn import manual_seed, initial_seed

manual_seed(12)
ms_seed = initial_seed()
print(ms_seed)
# 12
```
