# Differences with torch.deg2rad

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/deg2rad.md)

## torch.deg2rad

```text
torch.deg2rad(input, *, out=None) -> Tensor
```

For more information, see [torch.deg2rad](https://pytorch.org/docs/1.8.1/generated/torch.deg2rad.html).

## mindspore.ops.deg2rad

```text
mindspore.ops.deg2rad(x) -> Tensor
```

For more information, see [mindspore.ops.deg2rad](https://www.mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.deg2rad.html).

## Differences

MindSpore API function is consistent with that of PyTorch, with differences in the data types supported by the parameters.

PyTorch: The dtype of the parameter `input` can be ``int`` or ``float``.

MindSpore: The dtype of the parameter `x` can be ``float16``, ``float32``, or ``float64``.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | input | x | The parameter names are different. Both are Tensor, and the dtype of the parameter `input` can be ``int`` or ``float``. The dtype of the parameter `x` can be ``float16``, ``float32`` or ``float64``.|
|      | Parameter 2 | out | - | For detailed, refer to [General Difference Parameter Table](https://www.mindspore.cn/docs/en/r2.1/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table). |