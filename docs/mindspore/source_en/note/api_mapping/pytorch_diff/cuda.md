# Function Differences with torch.cuda.set_device

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/cuda.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.Tensor.cuda

```python
torch.Tensor.cuda()
```

For more information, see [torch.Tensor.cuda](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.cuda).

## mindspore.set_context

```python
mindspore.set_context(**kwargs)
```

For more information, see [mindspore.set_context](https://mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_context.html#mindspore.set_context).

## Differences

PyTorch: Copy Tenosr to cuda memory.

MindSpore：When setting the variable to `device_target='GPU'` or `device_target='Ascend'`, both the network and Tensor will be automatically copied to the GPU/Ascend device.
