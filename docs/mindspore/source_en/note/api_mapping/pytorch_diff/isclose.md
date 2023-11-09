# Differences with torch.isclose

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/isclose.md)

## torch.isclose

```text
torch.isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor
```

For more information, see [torch.isclose](https://pytorch.org/docs/1.8.1/generated/torch.isclose.html)。

## mindspore.ops.isclose

```text
mindspore.ops.isclose(x1, x2, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor
```

For more information, see [mindspore.ops.isclose](https://www.mindspore.cn/docs/en/r2.3/api_python/ops/mindspore.ops.isclose.html)。

## Differences

API function of MindSpore is consistent with that of PyTorch, with differences in the supported data types for parameters.

PyTorch: The dtype of parameter `input` and `other` can be ``bool`` , ``int`` or ``float`` .

MindSpore： The dtype of parameter `x1` and `x2` can be ``int32`` , ``float32`` or ``float16`` .

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | input | x1 | Different parameter names. The parameter `input` and `x1` are Tensors，while the dtype of `input` can be ``bool`` , ``int`` or ``float`` , the dtype of `x1` can be ``int32`` , ``float32`` or ``float16`` . |
|  | Parameter 2 | other | x2 | Different parameter names. The parameter `other` and `x2` are Tensors，while the dtype of `other` can be ``bool`` , ``int`` or ``float`` , the dtype of `x2` can be ``int32`` , ``float32`` or ``float16`` . |
|  | Parameter 3 | rtol | rtol | - |
|  | Parameter 4 | atol | atol | - |
|  | Parameter 5 | equal_nan | equal_nan | - |
