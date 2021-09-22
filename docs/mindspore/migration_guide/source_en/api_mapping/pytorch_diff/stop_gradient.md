# Comparing the functional differences with torch.autograd.enable_grad and torch.autograd.no_grad

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/stop_gradient.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## torch.autograd.enable_grad

```python
torch.autograd.enable_grad()
```

## torch.autograd.no_grad

```python
torch.autograd.no_grad()
```

## mindspore.ops.stop_gradient

```python
mindspore.ops.stop_gradient(input)
```

## Differences

PyTorch: Use `torch.autograd.enable_grad` to enable gradient calculation, and `torch.autograd.no_grad` to disable gradient calculation.

MindSpore: Use [stop_gradient](https://www.mindspore.cn/tutorials/en/master/autograd.html#stop-gradient) to disable calculation of gradient for certain operators.
