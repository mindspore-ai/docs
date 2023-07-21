# 比较与torch.cuda.set_device的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.8/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/set_context.md)

## torch.cuda.set_device

```python
torch.cuda.set_device(device)
```

更多内容详见[torch.cuda.set_device](https://pytorch.org/docs/1.5.0/cuda.html#torch.cuda.set_device)。

## mindspore.set_context

```python
mindspore.set_context(**kwargs)
```

更多内容详见[mindspore.set_context](https://mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.set_context.html#mindspore.set_context)。

## 使用方式

PyTorch：设置当前使用的`device`卡号。

MindSpore：不仅设置当前使用的`device`卡号，还设置模式`mode`，运行环境`device_target`，是否保存图`save_graphs`等。