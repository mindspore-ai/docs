# Differences with torch.distributed.all_gather

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/all_gather.md)

## torch.distributed.all_gather

```python
torch.distributed.all_gather(
    tensor_list,
    tensor,
    group=None,
    async_op=False
)
```

For more information, see [torch.distributed.all_gather](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.all_gather).

## mindspore.communication.comm_func.all_gather_into_tensor

```python
from mindspore.communication.comm_func import all_gather_into_tensor
return_tensor = all_gather_into_tensor(
    tensor,
    group=None,
    async_op=False
)
```

For more information, see [mindspore.communication.comm_func.all_gather_into_tensor](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.comm_func.all_gather_into_tensor.html#mindspore.communication.comm_func.all_gather_into_tensor).

## Differences

PyTorch: The inputs are the tensor broadcasted by the current process `tensor`, the communication group `group` and the async op flag `async_op`. The output is `tensor_list` after all_gather op, whose type is `list[Tensor]` and the length is the number of devices in the communication group. The return is a async work handle if `async_op=True`, otherwise is `None`.

MindSpore: This interface has two inputs, the input tensor `tensor`, the communication group `group` and the async op flag `async_op`. The first dimension of the input `tensor` is the number of devices N in the communication domain, and the rest of the dimensions is the same as the input tensor, rather than outputting list[Tensor] as the PyTorch interface does.

| Class      | Sub-class     |PyTorch | MindSpore | Difference                                                                                 |
|------------|---------------| --- |-----------|--------------------------------------------------------------------------------------------|
| Parameters | Parameter 1   | tensor_list | -         | PyTorch: the output after all_gather. MindSpore does not have this parameter.               |
|            | Parameter 2   | tensor | tensor    | No difference                                                                              |
|            | Parameter 3   | group | group     | No difference                                                                              |
|            | Parameter 4   | async_op | async_op      |             No difference            |
| Returns    | Single return | - | tensor    | PyTorch: does not have a return. MindSpore: returns the output tensor after all_gather_into_tensor Operation. |
