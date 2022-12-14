# Function Differences with torch.cuda.set_device

<a href="https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_en/note/api_mapping/pytorch_diff/cuda.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source_en.png"></a>

## torch.Tensor.cuda

```python
torch.Tensor.cuda()
```

For more information, see [torch.Tensor.cuda](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.cuda).

## mindspore.set_context

```python
mindspore.set_context(**kwargs)
```

For more information, see [mindspore.set_context](https://mindspore.cn/docs/en/r1.10/api_python/mindspore/mindspore.set_context.html#mindspore.set_context).

## Differences

PyTorch: It is used to copy object in CUDA memory.

MindSpore：When set parameter `device_target='GPU'` or `device_target='Ascend'`, Network and Tensor are copied to GPU/Ascend device automatically.
