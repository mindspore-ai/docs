# Differences with torch.default_generator

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/default_generator.md)

## torch.default_generator

```text
class torch.default_generator
```

For more information, see [torch.default_generator](https://pytorch.org/docs/1.8.1/torch.html#torch.default_generator).

## mindspore.nn.default_generator

```text
class mindspore.nn.default_generator
```

For more information, see [mindspore.nn.default_generator](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.default_generator.html).

## Differences

PyTorch: `default_generator` is used to manage the default generator. When the user does not specify a generator, the random operator calls the default generator to generate random numbers.

MindSpore: The implementation function of API in MindSpore is basically the same as that of PyTorch. MindSpore returns `mindspore.nn.Generator` objects, while PyTorch returns c++ side objects.

### Code Example 1

> The two APIs implement the same function and have the same usage.

```python
#PyTorch
import torch

torch_gen = torch.default_generator
print(type(torch_gen))
# <class 'torch._C.Generator'>


# MindSpore
from mindspore.nn import default_generator

ms_gen = default_generator()
print(type(ms_gen))
# <class 'mindspore.nn.generator.Generator'>
```
