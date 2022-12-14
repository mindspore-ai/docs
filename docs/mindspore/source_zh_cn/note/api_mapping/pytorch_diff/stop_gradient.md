# 比较与torch.autograd.enable_grad和torch.autograd.no_grad的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/stop_gradient.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source.png"></a>

## torch.autograd.enable_grad

```python
torch.autograd.enable_grad()
```

更多内容详见[torch.autograd.enable_grad](https://pytorch.org/docs/1.5.0/autograd.html#torch.autograd.enable_grad)。

## torch.autograd.no_grad

```python
torch.autograd.no_grad()
```

更多内容详见[torch.autograd.no_grad](https://pytorch.org/docs/1.5.0/autograd.html#torch.autograd.no_grad)。

## mindspore.ops.stop_gradient

```python
mindspore.ops.stop_gradient(input)
```

更多内容详见[mindspore.ops.stop_gradient](https://www.mindspore.cn/tutorials/zh-CN/r1.10/beginner/autograd.html#stop-gradient)。

## 使用方式

PyTorch：使用`torch.autograd.enable_grad`启用梯度计算，使用`torch.autograd.no_grad`禁用梯度计算。

MindSpore：使用[stop_gradient](https://www.mindspore.cn/tutorials/zh-CN/r1.10/beginner/autograd.html#stop-gradient)禁止网络内的算子对梯度的影响。
