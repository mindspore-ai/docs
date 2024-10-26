# 比较与torch.distributed.reduce_scatter的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/reduce_scatter_tensor.md)

## torch.distributed.reduce_scatter

```python
torch.distributed.reduce_scatter(
    output,
    input_list,
    op=ReduceOp.SUM,
    group=None,
    async_op=False
)
```

更多内容详见[torch.distributed.reduce_scatter](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.reduce_scatter)。

## mindspore.communication.comm_func.reduce_scatter_tensor

```python
mindspore.communication.comm_func.reduce_scatter_tensor(
    tensor,
    op=ReduceOp.SUM,
    group=None,
    async_op=False
)
```

更多内容详见[mindspore.communication.comm_func.reduce_scatter_tensor](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/communication/mindspore.communication.comm_func.reduce_scatter_tensor.html#mindspore.communication.comm_func.reduce_scatter_tensor)。

## 差异对比

PyTorch：该接口输入`output`将保存分散结果，`input_list`包含要reduce_scatter的张量列表。每个进程提供相同数量的张量（具有相同的大小）。`group`是通信组，异步操作标志`async_op`。如果`async_op=True`，则返回一个异步工作句柄，否则返回`None`。

MindSpore：该接口输入张量`tensor`，通信组`group`和异步操作标志`async_op`，异步操作标志`async_op`。输入`tensor`的第一个维度可以被N（通信域中的设备数量）整除，其余维度与输入张量相同。如果`async_op=True`，则返回一个异步工作句柄，否则返回`None`。

| 分类 | 子类   | PyTorch    | MindSpore | 差异                                                                                                                                                                      |
|----|------|------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 参数 | 参数1  | output     | -         | PyTorch：输入tensor，进行reduce_scatter操作后将结果写回tensor，MindSpore无此参数                                                                                                           |
|    | 参数2  | input_list | tensor    | PyTorch：“input_list”包含需要被reduce_scatter的张量列表，MindSpore：输入tensor的第一个维度可以被N（通信域中的设备数量）整除，其余维度与输入张量相同。                                                                     |
|    | 参数3  | op         | op        | 一致                                                                                                                                                                      |
|    | 参数4  | group      | group     | 一致                                                                                                                                                                      |
|    | 参数5  | async_op   | async_op         | 一致                                                                                                                                                                      |
| 返回值 | 单返回值 | async_work_handle    | tuple(tensor, CommHandle)    | PyTorch：如果`async_op`设置为 True，返回异步工作句柄。如果`async_op`为False或不是组的一部分，则为None。</br> MindSpore：返回一个元组。该元组包含了reduce_scatter_tensor操作后返回的张量以及一个异步返回句柄。当`async_op`为False，该异步返回句柄为None。 |
