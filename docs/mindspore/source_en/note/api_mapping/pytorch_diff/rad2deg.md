# Function differences with torch.rad2deg

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/rad2deg.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.rad2deg

```text
torch.rad2deg(input, *, out=None) -> Tensor
```

For more information, see [torch.rad2deg](https://pytorch.org/docs/1.8.1/generated/torch.rad2deg.html).

## mindspore.ops.rad2deg

```text
mindspore.ops.rad2deg(x) -> Tensor
```

For more information, see [mindspore.ops.rad2deg](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.rad2deg.html).

## Differences

MindSpore API function is consistent with that of PyTorch, with differences in the data types supported by the parameters.

PyTorch: The dtype of the parameter `input` can be ``int`` or ``float``.

MindSpore: The dtype of the parameter `x` can be ``float16``, ``float32`` or ``float64``.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | input | x | The parameter names are different. Both are Tensor, and the dtype of the parameter `input` can be ``int`` or ``float``. The dtype of the parameter `x` can be ``float16``, ``float32``, ``float64``.|
|      | Parameter 2 | out | - | For detailed, refer to [General Difference Parameter Table](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table). |
