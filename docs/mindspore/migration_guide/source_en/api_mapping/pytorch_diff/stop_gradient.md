# Comparing the functional differences with torch.autograd.enable_grad and torch.autograd.no_grad

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/stop_gradient.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## torch.autograd.enable_grad

```python
torch.autograd.enable_grad()
```

For more information, see[torch.autograd.enable_grad](https://pytorch.org/docs/1.5.0/autograd.html#torch.autograd.enable_grad).

## torch.autograd.no_grad

```python
torch.autograd.no_grad()
```

For more information, see[torch.autograd.no_grad](https://pytorch.org/docs/1.5.0/autograd.html#torch.autograd.no_grad).

## mindspore.ops.stop_gradient

```python
mindspore.ops.stop_gradient(input)
```

For more information, see[mindspore.ops.stop_gradient](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.ops.html#functional).

## Differences

PyTorch: Use `torch.autograd.enable_grad` to enable gradient calculation, and `torch.autograd.no_grad` to disable gradient calculation.

MindSpore: Use [stop_gradient](https://www.mindspore.cn/tutorials/en/master/autograd.html#stop-gradient) to disable calculation of gradient for certain operators.
