# Differences with torch.polygamma

<a href="https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/polygamma.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png"></a>

## torch.polygamma

```text
torch.polygamma(n, input, *, out=None) -> Tensor
```

For more information, see [torch.polygamma](https://pytorch.org/docs/1.8.1/generated/torch.polygamma.html).

## mindspore.ops.polygamma

```text
mindspore.ops.polygamma(n, input) -> Tensor
```

For more information, see [mindspore.ops.polygamma](https://www.mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.polygamma.html).

## Differences

MindSpore API function is consistent with that of PyTorch, with differences in the data types supported by the parameters.

PyTorch: The dtype of the parameter `input` can be ``int`` or ``float``.

MindSpore: The dtype of the parameter `input` can be ``float16``, ``float32`` or ``float64``.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | n | n | - |
|  | Parameter 2 | input | input | Both are Tensor, and the dtype of the parameter `input` in torch.polygamma can be ``int`` or ``float``. The dtype of the parameter `input` in mindspore.ops.polygamma can be ``float16``, ``float32`` or ``float64``.|
|      | Parameter 3 | out | - | For detailed, refer to [General Difference Parameter Table](https://www.mindspore.cn/docs/en/r2.1/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table). |
