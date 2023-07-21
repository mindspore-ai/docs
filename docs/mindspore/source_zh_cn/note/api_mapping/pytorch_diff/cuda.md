# 比较与 torch.Tensor.cuda 的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/cuda.md)

## torch.Tensor.cuda

```python
torch.Tensor.cuda()
```

更多内容详见 [torch.Tensor.cuda](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.cuda).

## mindspore.set_context

```python
mindspore.set_context(**kwargs)
```

更多内容详见 [mindspore.set_context](https://mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore/mindspore.set_context.html#mindspore.set_context).

## Differences

PyTorch: 将Tenosr 拷贝到 cuda 内存.

MindSpore：将变量设置为 `device_target='GPU'` 或 `device_target='Ascend'` 时, 网络和Tensor都将自动拷贝到 GPU/Ascend 设备.
