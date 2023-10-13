# Differences with torch.mvlgamma

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_en/note/api_mapping/pytorch_diff/mvlgamma.md)

## torch.mvlgamma

```text
torch.mvlgamma(input, p) -> Tensor
```

For more information, see [torch.mvlgamma](https://pytorch.org/docs/1.8.1/generated/torch.mvlgamma.html).

## mindspore.ops.mvlgamma

```text
mindspore.ops.mvlgamma(input, p) -> Tensor
```

For more information, see [mindspore.ops.mvlgamma](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.mvlgamma.html).

## Differences

MindSpore API function is consistent with that of PyTorch, with differences in the data types supported by the parameters.

PyTorch: The dtype of the parameter `input` can be ``int`` or ``float``.

MindSpore: The dtype of the parameter `input` can be ``float32`` or ``float64``.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | input | input | Both are Tensor, and the dtype of the parameter `input` in torch.mvlgamma can be ``int`` or ``float``. The dtype of the parameter `input` in mindspore.ops.mvlgamma can be ``float32`` or ``float64``.|
|      | Parameter 2 | p | p | - |