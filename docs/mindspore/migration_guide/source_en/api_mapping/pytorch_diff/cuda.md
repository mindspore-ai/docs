﻿# Function Differences with torch.cuda.set_device

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/cuda.md)

## torch.Tensor.cuda

```python
torch.Tensor.cuda()
```

For more information, see [torch.Tensor.cuda](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.cuda).

## mindspore.context.set_context

```python
mindspore.context.set_context(**kwargs)
```

For more information, see [mindspore.context.set_context](https://mindspore.cn/docs/api/en/r1.6/api_python/mindspore.context.html#mindspore.context.set_context).

## Differences

PyTorch: It is used to copy object in CUDA memory.

MindSpore：When set parameter `device_target='GPU'` or `device_target='Ascend'`, Network and Tensor are copied to GPU/Ascend device automatically.
