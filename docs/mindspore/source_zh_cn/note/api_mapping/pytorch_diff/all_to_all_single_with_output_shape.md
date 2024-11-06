# 比较与torch.distributed.all_to_all_single的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/all_to_all_single_with_output_shape.md)

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

更多内容详见[torch.distributed.all_to_all_single](https://pytorch.org/docs/2.0/distributed.html#torch.distributed.all_to_all_single)。

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

更多内容详见[mindspore.communication.comm_func.all_to_all_single_with_output_shape](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/communication/mindspore.communication.comm_func.all_to_all_single_with_output_shape.html#mindspore.communication.comm_func.all_to_all_single_with_output_shape)。

## 使用方式

PyTorch：该接口传入待接收的张量、待切分发送的张量、接收张量的切分列表、发送张量的接收列表、通信组以及是否异步执行的参数，将接收到数据原地修改到output入参中。如果开启异步执行，则会返回该一个handler，以便后续做同步操作。

MindSpore：该接口传入待接收的张量的形状、待切分发送的张量、接收张量的切分列表、发送张量的接收列表以及通信组，通过返回值的方式返回接收到的数据。如果开启异步执行，则会返回一个元组，其中包含接收到的数据和handler；如果未开启异步执行，元组中则包含返回接收的数据和None。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | output | output_shape | 功能不一致，类型不同。 PyTorch传入待接收数据的张量，结果通过原地赋值的方式赋予传入的张量；MindSpore传入待接收数据的张量形状，结果并通过一个新的张量返回。|
| | 参数2 | input | tensor | 功能一致，参数名称不同 |
| | 参数3 | output_split_sizes | output_split_sizes | 功能一致 |
| | 参数4 | input_split_sizes | input_split_sizes | 功能一致 |
| | 参数5 | group | group | 功能一致，类型不同。 PyTorch传入的通信组对象； MindSpore传入通信组的字符串名称。 |
| | 参数6 | async_op | async_op | 功能一致 |
