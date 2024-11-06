# 比较与torch.distributed.all_to_all的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/all_to_all_with_output_shape.md)

## torch.distributed.all_to_all

```python
torch.distributed.all_to_all(
    output_tensor_list,
    input_tensor_list,
    group=None,
    async_op=False)
```

更多内容详见[torch.distributed.all_to_all](https://pytorch.org/docs/2.0/distributed.html#torch.distributed.all_to_all)。

## mindspore.communication.comm_func.all_to_all_with_output_shape

```python
mindspore.communication.comm_func.all_to_all_with_output_shape(
    output_shape_list,
    input_tensor_list,
    group=None
    async_op=False
)
```

更多内容详见[mindspore.communication.comm_func.all_to_all_with_output_shape](https://www.mindspore.cn/docs/zh-CN/master/api_python/communication/mindspore.communication.comm_func.all_to_all_with_output_shape.html#mindspore.communication.comm_func.all_to_all_with_output_shape)。

## 使用方式

PyTorch：该接口传入待接收的张量列表、待发送的张量列表、通信组以及是否异步执行的参数，将接收到数据原地修改到output_tensor_list入参的张量中。如果开启异步执行，则会返回该一个handler，以便后续做同步操作。

MindSpore：该接口传入待接收的张量的形状列表、待发送的张量列表表以及通信组，通过返回值的方式返回接收到的张量元组。如果开启异步执行，则会返回一个元组，其中包含接收到的数据和handler；如果未开启异步执行，元组中则包含返回接收的数据和None。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | output_tensor_list | output_shape_list | 功能不一致，类型不同。PyTorch传入待接收数据的张量列表，接收到的数据原地赋值给传入的张量列表；MindSpore传入待接收数据的张量形状列表，接收到的数据通过新的张量列表返回。|
| | 参数2 | input_tensor_list | input_tensor_list | 功能一致。 |
| | 参数3 | group | group | 功能一致，类型不同。PyTorch传入的通信组对象； MindSpore传入通信组的字符串名称。 |
| | 参数4 | async_op | async_op | 功能一致 |
