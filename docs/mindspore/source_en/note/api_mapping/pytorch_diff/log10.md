# Differences with torch.log10

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_en/note/api_mapping/pytorch_diff/log10.md)

## torch.log10

```text
torch.log10(input, *, out=None) -> Tensor
```

For more information, see [torch.log10](https://pytorch.org/docs/1.8.1/generated/torch.log10.html).

## mindspore.ops.log10

```text
mindspore.ops.log10(input) -> Tensor
```

For more information, see [mindspore.ops.log10](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.log10.html).

## Differences

MindSpore API function is consistent with that of PyTorch, with differences in the data types supported by the parameters.

PyTorch: The dtype of the parameter `input` can be ``int`` or ``float``.

MindSpore: The dtype of the parameter `input` can be ``float16``, ``float32`` or ``float64`` on GPU and CPU platforms, while the dtype of the parameter `input` can be ``float16`` or ``float32`` on Ascend platform.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | input | input | Both are Tensor, and the dtype of the parameter `input` in torch.log10 can be ``int`` or ``float``. The dtype of the parameter `input` in mindspore.ops.log10 can be ``float16``, ``float32``, ``float64`` on GPU and CPU platforms, while ``float16`` or ``float32`` on Ascend platform.|
|      | Parameter 2 | out | - | For detailed, refer to [General Difference Parameter Table](https://www.mindspore.cn/docs/en/r2.2/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table). |