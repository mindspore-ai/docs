# 比较与torch.distributed.reduce的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/reduce.md)

## torch.distributed.reduce

```python
torch.distributed.reduce(
    tensor,
    dst,
    op=ReduceOp.SUM,
    group=None,
    async_op=False
)
```

更多内容详见[torch.distributed.reduce](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.reduce)。

## mindspore.communication.comm_func.reduce

```python
from mindspore.communication.comm_func import reduce
return_tensor = reduce(
    tensor,
    dst,
    op=ReduceOp.SUM,
    group=None,
)
```

更多内容详见[mindspore.communication.comm_func.reduce](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.comm_func.reduce.html#mindspore.communication.comm_func.reduce)。

## 差异对比

PyTorch：该接口有五个入参：“tensor”将存储目标进程“dst”的reduce结果，并且所有张量必须具有相同的形状。只有“dst”进程会存储reduce后的结果，其他进程的张量保持不变。“group”是通信组，异步操作标志“async_op”。如果“async_op=True”，则返回一个异步工作句柄，否则返回“None”。

MindSpore：该接口有四个入参和一个返回值，每个进程的输入“tensor”必须具有相同的形状，并且不会存储reduce的结果，通信组“group”和“op”与pytorch相同。该接口目前不支持`async_op`的配置。返回的“output”是一个张量，存储目标进程的reduce的结果。只有“dst”进程的返回值才会存储reduce的结果，其他进程的返回值保持零张量，与输入张量的形状相同。

| 分类 | 子类   | PyTorch  | MindSpore | 差异                                                                     |
|----|------|----------|-----------|------------------------------------------------------------------------|
| 参数 | 参数1  | tensor   | tensor         | PyTorch：输入tensor，进行reduce操作后将结果写回tensor，MindSpore:不会将reduce的结果写回tensor |
| 参数 | 参数2  | dst      | dst       | 一致                                                                     |
|    | 参数3  | op       | op        | 一致                                                                     |
|    | 参数4  | group    | group     | 一致                                                                     |
|    | 参数5  | async_op | -         | PyTorch：异步操作标志，MindSpore无此参数                                           |
| 返回值 | 单返回值 | -        | tensor    | PyTorch：没有返回值。 MindSpore：返回reduce操作后返回的张量。                             |
