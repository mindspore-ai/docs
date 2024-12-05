# Differences with torch.distributed.all_gather

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.10/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.10/docs/mindspore/source_en/note/api_mapping/pytorch_diff/all_gather.md)

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
mindspore.communication.comm_func.all_gather_into_tensor(
    tensor,
    group=None,
    async_op=False
)
```

For more information, see [mindspore.communication.comm_func.all_gather_into_tensor](https://www.mindspore.cn/docs/en/r2.4.10/api_python/communication/mindspore.communication.comm_func.all_gather_into_tensor.html#mindspore.communication.comm_func.all_gather_into_tensor).

## Differences

PyTorch: The inputs of this interface are the input tensor `tensor`, the communication group `group` and the async op flag `async_op`. The result after all_gather op is `tensor_list`, whose type is `list[Tensor]` and the length is the number of devices in the communication group. The return is an async work handle if `async_op=True`, otherwise is `None`.

MindSpore: The inputs of this interface are the input tensor `tensor`, the communication group `group` and the async op flag `async_op`. The first dimension of the output `tensor` is the number of devices N in the communication domain, and the rest of the dimensions is the same as the input tensor, rather than outputting list[Tensor] as the PyTorch interface does. Another return is an async work handle if `async_op=True`, otherwise is `None`.

| Class      | Sub-class     | PyTorch           | MindSpore                 | Difference                                                                                                                                                                                                                                                                                                           |
|------------|---------------|-------------------|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Parameters | Parameter 1   | tensor_list       | -                         | PyTorch: the output after all_gather. MindSpore does not have this parameter.                                                                                                                                                                                                                                        |
|            | Parameter 2   | tensor            | tensor                    | No difference                                                                                                                                                                                                                                                                                                        |
|            | Parameter 3   | group             | group                     | No difference                                                                                                                                                                                                                                                                                                        |
|            | Parameter 4   | async_op          | async_op                  | No difference                                                                                                                                                                                                                                                                                                        |
| Returns    | Single return | async_work_handle | tuple(tensor, CommHandle) | PyTorch: An async work handle, if async_op is set to True. None, if not async_op or if not part of the group.</br> MindSpore: returns a tuple. The tuple contains an output tensor after all_gather_into_tensor operation and an async work handle `CommHandle`. When `async_op` is False, the `CommHandle` will be None. |
