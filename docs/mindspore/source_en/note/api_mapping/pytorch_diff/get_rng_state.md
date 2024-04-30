# Differences with torch.get_rng_state

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/get_rng_state.md)

## torch.get_rng_state

```text
class torch.get_rng_state
```

For more information, see [torch.get_rng_state](https://pytorch.org/docs/1.8.1/generated/torch.get_rng_state.html).

## mindspore.nn.get_rng_state

```text
class mindspore.nn.get_rng_state
```

For more information, see [mindspore.nn.get_rng_state](https://www.mindspore.cn/docs/en/r2.3/api_python/nn/mindspore.nn.get_rng_state.html).

## Differences

PyTorch: `get_rng_state` returns the state of the default generator. The generator uses a Tensor of type `ByteTensor` to save the state. The generator state contains the values `seed` and `offset`, where `seed` provides the seed for the random operator to generate a random sequence and `offset` provides the offset on the random sequence. When `seed` and `offset` are fixed, the randomizer generates fixed values.

MindSpore: The implementation function of API in MindSpore is basically the same as that of PyTorch. MindSpore stores `seed` , `offset` separately, so the MindSpore generator state returns a tuple of Tensor containing `seed` and `offset` values.

### Code Example 1

> The two APIs implement the same function and have the same usage.

```python
#PyTorch
import torch

torch.manual_seed(12)
torch_state = torch.get_rng_state()
print(torch_state)
# tensor([12,  0,  0,  ...,  0,  0,  0], dtype=torch.uint8)


# MindSpore
from mindspore.nn import manual_seed, get_rng_state

manual_seed(12)
ms_seed, ms_offset = get_rng_state()
print(ms_seed)
# 12
print(ms_offset)
# 0
```
