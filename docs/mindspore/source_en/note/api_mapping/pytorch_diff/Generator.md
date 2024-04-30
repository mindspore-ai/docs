# Differences with torch.Generator

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3.0rc2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3.0rc2/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Generator.md)

## torch.Generator

```text
class torch.Generator(device='cpu')
```

For more information, see [torch.Generator](https://pytorch.org/docs/1.8.1/generated/torch.Generator.html).

## mindspore.nn.Generator

```text
class mindspore.nn.Generator()
```

For more information, see [mindspore.nn.Generator](https://www.mindspore.cn/docs/en/r2.3.0rc2/api_python/nn/mindspore.nn.Generator.html).

## Differences

PyTorch: The `Generator` is the generator that manages the random state, using a Tensor of type `ByteTensor` to hold the generator state. The generator state contains the values `seed` and `offset`, where `seed` provides the random operator with the seed for generating the random sequence and `offset` provides the offset on the random sequence. When `seed` and `offset` are fixed, the randomizer generates fixed values.

MindSpore: The implementation function of API in MindSpore is basically the same as that of PyTorch, but there are differences in usage. MindSpore's generator for managing random states is implemented on the python side, inheriting from `mindspore.nn.Cell`. PyTorch is implemented on the c++ side, with the interface wrapped on the python side. PyTorch uses `ByteTensor` to store `seed` and `offset` in a Tensor, while MindSpore stores `seed` , `offset` separately. PyTorch updates `offset` on the c++ side, and MindSpore rewrites `construct` method in `mindspore.nn.Cell` to update `offset`.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|   Parameter   | Parameter 1 |    device     | - | PyTorch sets the backend of the generator, and MindSpore does not have this parameter |
| Input  | Single Input | -      | step           | PyTorch updates `offset` on the backend, MindSpore adds `step` value to `offset`     |
| Method | - | device | - |  PyTorch returns the backend of the generator, MindSpore does not have this method  |
| Method | - | get_state | get_state |  PyTorch returns values of type `ByteTensor` and MindSpore returns tuples of Tensor containing `seed` and `offset` values. |
| Method | - | initial_seed | initial_seed |  PyTorch returns `seed` of type int and MindSpore returns `seed` of type Tensor. |
| Method | - | manual_seed | manual_seed | Consistent  |
| Method | - | seed | seed |  PyTorch randomly generates `seed` of type int via c++ backend, and MindSpore randomly generates `seed` of type Tensor. MindSpore generates random seeds via numpy, and fixing numpy's randomness fixes the randomness of this interface |
| Method | - | set_state | set_state |  PyTorch passes in the state saved by the `ByteTensor` type and MindSpore passes in the state saved by `seed` and `offset`. |

### Code Example 1

> The two APIs implement the same function and have the same usage.

```python
#PyTorch
import torch

torch_gen = torch.Generator(device="cpu")
print(torch_gen.device)
# cpu
torch_gen.manual_seed(12)
print(torch_gen.initial_seed())
# 12
torch_state = torch_gen.get_state()
print(torch_state)
# tensor([12,  0,  0,  ...,  0,  0,  0], dtype=torch.uint8)
torch_gen.seed()
torch_gen.set_state(torch_state)
print(torch_gen.get_state())
# tensor([12,  0,  0,  ...,  0,  0,  0], dtype=torch.uint8)


# MindSpore
from mindspore.nn import Generator

ms_gen = Generator()
ms_gen.manual_seed(12)
print(ms_gen.initial_seed())
# 12
ms_seed, ms_offset = ms_gen.get_state()
print(ms_seed)
# 12
print(ms_offset)
# 0
ms_gen.seed()
ms_gen.set_state(ms_seed, ms_offset)
print(ms_gen.get_state())
#(Tensor(shape=[], dtype=Int32, value= 12), Tensor(shape=[], dtype=Int32, value= 0))
```