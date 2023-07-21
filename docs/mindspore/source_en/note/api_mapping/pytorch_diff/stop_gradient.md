# Comparing the functional differences with torch.autograd.enable_grad and torch.autograd.no_grad

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/stop_gradient.md)

## torch.autograd.enable_grad

```python
torch.autograd.enable_grad()
```

For more information, see [torch.autograd.enable_grad](https://pytorch.org/docs/1.5.0/autograd.html#torch.autograd.enable_grad).

## torch.autograd.no_grad

```python
torch.autograd.no_grad()
```

For more information, see [torch.autograd.no_grad](https://pytorch.org/docs/1.5.0/autograd.html#torch.autograd.no_grad).

## mindspore.ops.stop_gradient

```python
mindspore.ops.stop_gradient(input)
```

For more information, see [mindspore.ops.stop_gradient](https://mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.stop_gradient.html).

## Differences

PyTorch: Use `torch.autograd.enable_grad` to enable gradient calculation, and `torch.autograd.no_grad` to disable gradient calculation.

MindSpore: Use [stop_gradient](https://mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.stop_gradient.html) to disable calculation of gradient for certain operators.
