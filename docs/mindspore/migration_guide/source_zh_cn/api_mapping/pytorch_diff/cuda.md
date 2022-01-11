﻿# 比较与 torch.Tensor.cuda 的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/cuda.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## torch.Tensor.cuda

```python
torch.Tensor.cuda()
```

更多内容详见 [torch.Tensor.cuda](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.cuda).

## mindspore.context.set_context

```python
mindspore.context.set_context(**kwargs)
```

更多内容详见 [mindspore.context.set_context](https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore.context.html#mindspore.context.set_context).

## Differences

PyTorch: 将Tenosr 拷贝到 cuda 内存.

MindSpore：将变量设置为 `device_target='GPU'` 或 `device_target='Ascend'` 时, 网络和Tensor都将自动拷贝到 GPU/Ascend 设备.
