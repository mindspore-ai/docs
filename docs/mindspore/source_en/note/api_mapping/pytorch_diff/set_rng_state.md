# Differences with torch.set_rng_state

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.set_rng_statemyhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/set_rng_state.md)

## torch.set_rng_state

```text
class torch.set_rng_state
```

For more information, see [torch.set_rng_state](https://pytorch.org/docs/1.8.1/generated/torch.set_rng_state.html).

## mindspore.nn.set_rng_state

```text
class mindspore.nn.set_rng_state
```

For more information, see [mindspore.nn.set_rng_state](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.set_rng_state.html).

## Differences

PyTorch: `set_rng_state` sets the state of the default generator. The generator uses a Tensor of type `ByteTensor` to save the state. The generator state contains the values `seed` and `offset`, where `seed` provides the seed for the random operator to generate a random sequence and `offset` provides the offset on the random sequence. When `seed` and `offset` are fixed, the randomizer generates fixed values.

MindSpore: The implementation function of API in MindSpore is basically the same as that of PyTorch. MindSpore stores `seed` and `offset` separately, so MindSpore generator state needs to set a Tensor with `seed` and `offset` values, and PyTorch needs to set a `ByteTensor` value that holds the generator state.

### Code Example 1

> The two APIs implement the same function and have the same usage.

```python
#PyTorch
import torch

torch.manual_seed(12)
torch_state = torch.get_rng_state()
torch.set_rng_state(torch_state)
torch_state = torch.get_rng_state()
print(torch_state)
# tensor([12,  0,  0,  ...,  0,  0,  0], dtype=torch.uint8)


# MindSpore
from mindspore.nn import manual_seed, set_rng_state, get_rng_state

manual_seed(12)
ms_seed, ms_offset = get_rng_state()
set_rng_state(ms_seed, ms_offset)
ms_seed, ms_offset = get_rng_state()
print(ms_seed)
# 12
print(ms_offset)
# 0
```
