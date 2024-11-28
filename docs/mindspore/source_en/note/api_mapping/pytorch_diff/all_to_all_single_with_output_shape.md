# Differences with torch.distributed.all_to_all_single

[![View Source](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/all_to_all_single_with_output_shape.md)

## torch.distributed.all_to_all_single

```python
torch.distributed.all_to_all_single(
    output,
    input,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
    async_op=False)
```

For more details, see [torch.distributed.all_to_all_single](https://pytorch.org/docs/2.0/distributed.html#torch.distributed.all_to_all_single).

## mindspore.communication.comm_func.all_to_all_single_with_output_shape

```python
mindspore.communication.comm_func.all_to_all_single_with_output_shape(
    output_shape,
    tensor,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None
    async_op=False
)
```

For more details, see [mindspore.communication.comm_func.all_to_all_single_with_output_shape](https://www.mindspore.cn/docs/zh-CN/r2.4.1/api_python/communication/mindspore.communication.comm_func.all_to_all_single_with_output_shape.html#mindspore.communication.comm_func.all_to_all_single_with_output_shape).

## Usage

PyTorch: This interface takes the tensor to be received, the tensor to be split and sent, the split list for the receiving tensor, the reception list for the sending tensor, the communication group, and the parameter for whether to execute asynchronously, modifying the received data in place to the `output` parameter. If asynchronous execution is enabled, a handler will be returned for subsequent synchronization operations.

MindSpore: This interface takes the shapes of the tensors to be received, the tensors to be split and sent, the split list for the receiving tensors, the receive list for the sending tensors, and the communication group. It returns the received data through the return value. If asynchronous execution is enabled, a tuple is returned, which includes the received data and a handler; if asynchronous execution is not enabled, the tuple contains the received data and None.

| Category | Subcategory | PyTorch | MindSpore | Difference |
| --- | --- | --- | --- | --- |
| Parameters | Parameter 1 | output | output_shape | Function is inconsistent, types are different. PyTorch passes in the tensor to receive data, and the result is assigned in place to the passed tensor; MindSpore passes in the shape of the tensor to receive data, and the result is returned as a new tensor. |
|  | Parameter 2 | input | tensor | Function is consistent, parameter name is different |
|  | Parameter 3 | output_split_sizes | output_split_sizes | Function is consistent |
|  | Parameter 4 | input_split_sizes | input_split_sizes | Function is consistent |
|  | Parameter 5 | group | group | Function is consistent, types are different. PyTorch passes in a communication group object; MindSpore passes in the string name of the communication group. |
|  | Parameter 6 | async_op | async_op | The functionality is consistent |
