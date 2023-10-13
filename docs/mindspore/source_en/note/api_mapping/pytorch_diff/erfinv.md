# Differences with torch.erfinv

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_en/note/api_mapping/pytorch_diff/erfinv.md)

## torch.erfinv

```text
torch.erfinv(input, *, out=None) -> Tensor
```

For more information, see [torch.erfinv](https://pytorch.org/docs/1.8.1/generated/torch.erfinv.html).

## mindspore.ops.erfinv

```text
mindspore.ops.erfinv(input) -> Tensor
```

For more information, see [mindspore.ops.erfinv](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.erfinv.html).

## Differences

MindSpore API function is consistent with that of PyTorch, with differences in the data types supported by the parameters.

PyTorch: The dtype of the parameter `input` can be ``int`` or ``float``.

MindSpore: The dtype of the parameter `input` can be ``float16``, ``float32``, ``float64``.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | input | input | Both are Tensor, and the dtype of the parameter `input` in torch.erfinv can be ``int`` or ``float``. The dtype of the parameter `input` in mindspore.ops.erfinv can be ``float16``, ``float32``, ``float64``.|
|      | Parameter 2 | out | - | For detailed, refer to [General Difference Parameter Table](https://www.mindspore.cn/docs/en/r2.2/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table). |