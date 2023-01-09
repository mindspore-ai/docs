# Function Differences with torch.cuda.set_device

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/set_context.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.cuda.set_device

```python
torch.cuda.set_device(device)
```

For more information, see [torch.cuda.set_device](https://pytorch.org/docs/1.5.0/cuda.html#torch.cuda.set_device).

## mindspore.set_context

```python
mindspore.set_context(**kwargs)
```

For more information, see [mindspore.set_context](https://mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_context.html#mindspore.set_context).

## Differences

PyTorch: It is used to set the current `device`.

MindSpore: It is not only used to set the current `device`, but also set the `mode`, running environment `device_target`, whether to save graphs `save_graphs`.