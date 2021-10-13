# 比较与torch.cuda.set_device的功能差异

## torch.cuda.set_device

```python
torch.cuda.set_device(device)
```

更多内容详见[torch.cuda.set_device](https://pytorch.org/docs/1.5.0/cuda.html#torch.cuda.set_device)。

## mindspore.context.set_context

```python
mindspore.context.set_context(**kwargs)
```

更多内容详见[mindspore.context.set_context](https://mindspore.cn/docs/api/zh-CN/r1.3/api_python/mindspore.context.html#mindspore.context.set_context)。

## 使用方式

PyTorch：设置当前使用的`device`卡号。

MindSpore：不仅设置当前使用的`device`卡号，还设置模式`mode`，运行环境`device_target`，是否保存图`save_graphs`等。