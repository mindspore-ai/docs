# 比较与torch.cuda.set_device的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/set_context.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## torch.cuda.set_device

```python
torch.cuda.set_device(device)
```

## mindspore.context.set_context

```python
mindspore.context.set_context(**kwargs)
```

## 使用方式

PyTorch: 设置当前使用的`device`卡号。

MindSpore：不仅设置当前使用的`device`卡号，还设置模式`mode`，运行环境`device_target`，是否保存图`save_graphs`等。