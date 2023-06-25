# Differences with torch.log

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/log.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.log

```text
torch.log(input, *, out=None) -> Tensor
```

For more information, see [torch.log](https://pytorch.org/docs/1.8.1/generated/torch.log.html).

## mindspore.ops.log

```text
mindspore.ops.log(input) -> Tensor
```

For more information, see [mindspore.ops.log](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.log.html).

## Differences

MindSpore API function is consistent with that of PyTorch, with differences in the data types supported by the parameters.

PyTorch: The dtype of the parameter `input` can be ``int`` or ``float``.

MindSpore: The dtype of the parameter `input` can be ``float16``, ``float32`` or ``float64`` on CPU platform, while the dtype of the parameter `input` can be ``float16`` or ``float32`` on Ascend platform.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | input | input | Both are Tensor, and the dtype of the parameter `input` in torch.log can be ``int`` or ``float``. The dtype of the parameter `input` in mindspore.ops.log can be ``float16``, ``float32``, ``float64`` on CPU platform, while ``float16`` or ``float32`` on Ascend platform.|
|      | Parameter 2 | out | - | For detailed, refer to [General Difference Parameter Table](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table). |