# 比较与torch.distributed.all_gather的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/all_gather.md)

## torch.distributed.all_gather

```python
torch.distributed.all_gather(
    tensor_list,
    tensor,
    group=None,
    async_op=False
)
```

更多内容详见[torch.distributed.all_gather](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.all_gather)。

## mindspore.communication.comm_func.all_gather_into_tensor

```python
import mindspore.communication as comm
return_tensor = comm.comm_func.all_gather_into_tensor(
    tensor,
    group=None,
    async_op=False
)
```

更多内容详见[mindspore.communication.comm_func.all_gather_into_tensor](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.comm_func.all_gather_into_tensor.html#mindspore.communication.comm_func.all_gather_into_tensor)。

## 差异对比

PyTorch：该接口输入当前进程广播的`tensor`、通信域`group`及异步操作标志`async_op`，进行AllGather操作后输出tensor_list，类型为list[Tensor]，长度为通信域中设备数量N。当async_op=True时，返回异步work句柄，否则返回为空。

MindSpore：该接口有两个输入，输入张量 `tensor` 和通信组 `group` 及异步操作标志`async_op`。输入`tensor`第一个维度是通信域中的设备数量N，其余维度与输入张量相同，而不是像PyTorch接口那样输出list[Tensor]。

| 分类 | 子类  |PyTorch | MindSpore | 差异                                                         |
| --- |-----| --- | --- |------------------------------------------------------------|
|参数 | 参数1 | tensor_list | - | PyTorch：进行all_gather操作后的输出，MindSpore无此参数。                  |
| | 参数2 | tensor | tensor | 一致                                                          |
| | 参数3 | group | group | 一致                                                          |
| | 参数4 | async_op | async_op | 一致                              |
|返回值| 单返回值 | - |tensor| PyTorch：没有返回值。 MindSpore：返回all_gather_into_tensor操作后返回的张量。 |
