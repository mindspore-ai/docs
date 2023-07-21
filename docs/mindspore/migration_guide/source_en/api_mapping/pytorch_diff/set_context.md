﻿# Function Differences with torch.cuda.set_device

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/set_context.md)

## torch.cuda.set_device

```python
torch.cuda.set_device(device)
```

For more information, see [torch.cuda.set_device](https://pytorch.org/docs/1.5.0/cuda.html#torch.cuda.set_device).

## mindspore.context.set_context

```python
mindspore.context.set_context(**kwargs)
```

For more information, see [mindspore.context.set_context](https://mindspore.cn/docs/api/en/r1.5/api_python/mindspore.context.html#mindspore.context.set_context).

## Differences

PyTorch: It is used to set the current `device`.

MindSpore：It is not only used to set the current `device`, but also set the `mode`, `device_target`, `save_graphs`, etc.