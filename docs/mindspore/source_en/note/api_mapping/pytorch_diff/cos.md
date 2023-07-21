# Differences with torch.cos

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/cos.md)

## torch.cos

```text
torch.cos(input, *, out=None) -> Tensor
```

For more information, see [torch.cos](https://pytorch.org/docs/1.8.1/generated/torch.cos.html).

## mindspore.ops.cos

```text
mindspore.ops.cos(input) -> Tensor
```

For more information, see [mindspore.ops.cos](https://www.mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.cos.html).

## Differences

MindSpore API function is consistent with that of PyTorch, with differences in the data types supported by the parameters.

PyTorch: The dtype of the parameter `input` can be ``complex``, ``int`` or ``float``.

MindSpore: The dtype of the parameter `input` can be ``float16``, ``float32``, ``float64``, ``complex64`` or ``complex128``.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | input | input | Both are Tensor, and the dtype of the parameter `input` in torch.cos can be ``complex``, ``int`` or ``float``. The dtype of the parameter `input` in mindspore.ops.cos can be ``float16``, ``float32``, ``float64``, ``complex64`` or ``complex128``.|
|      | Parameter 2 | out | - | For detailed, refer to [General Difference Parameter Table](https://www.mindspore.cn/docs/en/r2.1/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table). |