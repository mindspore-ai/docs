# Differences with torch.floor

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/floor.md)

## torch.floor

```text
torch.floor(input, *, out=None) -> Tensor
```

For more information, see [torch.floor](https://pytorch.org/docs/1.8.1/generated/torch.floor.html).

## mindspore.ops.floor

```text
mindspore.ops.floor(input) -> Tensor
```

For more information, see [mindspore.ops.floor](https://www.mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.floor.html).

## Differences

MindSpore API function is consistent with that of PyTorch, with differences in the data types supported by the parameters.

PyTorch: The dtype of the parameter `input` can be ``int`` or ``float``.

MindSpore: The dtype of the parameter `input` can be ``float16``, ``float32``, or ``float64``.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | input | input | Both are Tensor, and the dtype of the parameter `input` in torch.floor can be ``int`` or ``float``. The dtype of the parameter `input` in mindspore.ops.floor can be ``float16``, ``float32`` or ``float64``.|
|      | Parameter 2 | out | - | For detailed, refer to [General Difference Parameter Table](https://www.mindspore.cn/docs/en/r2.1/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table). |
