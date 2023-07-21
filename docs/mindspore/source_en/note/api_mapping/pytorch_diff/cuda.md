# Function Differences with torch.cuda.set_device

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_en/note/api_mapping/pytorch_diff/cuda.md)

## torch.Tensor.cuda

```python
torch.Tensor.cuda()
```

For more information, see [torch.Tensor.cuda](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.cuda).

## mindspore.set_context

```python
mindspore.set_context(**kwargs)
```

For more information, see [mindspore.set_context](https://mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.set_context.html#mindspore.set_context).

## Differences

PyTorch: It is used to copy object in CUDA memory.

MindSpore：When set parameter `device_target='GPU'` or `device_target='Ascend'`, Network and Tensor are copied to GPU/Ascend device automatically.
