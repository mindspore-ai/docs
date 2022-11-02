# 比较与torch.distributed.init_process_group的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/init.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.distributed.init_process_group

```python
torch.distributed.init_process_group(
    backend,
    init_method=None,
    timeout=datetime.timedelta(0, 1800),
    world_size=-1,
    rank=-1,
    store=None,
    group_name=''
)
```

更多内容详见[torch.distributed.init_process_group](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.init_process_group)。

## mindspore.communication.init

```python
mindspore.communication.init(backend_name=None)
```

更多内容详见[mindspore.communication.init](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.communication.html#mindspore.communication.init)。

## 使用方式

PyTorch：该接口支持的集合通信有3种：MPI、Gloo、NCCL。该接口在初始化`backend`的同时，还提供`world_size`、`rank`和`timeout`等内容的配置。

MindSpore：该接口当前仅支持2种集合通信：HCCL、NCCL。而`world_size`、`rank`和`timeout`等内容的配置并不在该接口中设置，调用该接口之前，需设置相应的环境变量。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | backend | backend_name | 功能一致，参数名有差异，且支持的集合通信后端有差异|
| | 参数2 | init_method | - | PyTorch：初始化方法，MindSpore无此参数|
| | 参数3 | timeout | - |PyTorch：超时阈值，MindSpore无此参数且需在调用该接口之前配置相应环境变量|
| | 参数4 | world_size | - |PyTorch：通信域内设备数量，MindSpore无此参数且需在调用该接口之前配置相应环境变量 |
| | 参数5 | rank | - |PyTorch：当前进行rank，MindSpore无此参数且需在调用该接口之前配置相应环境变量 |
| | 参数6 | store | - |PyTorch：储存key/value标志，MindSpore无此参数 |
| | 参数7 | group_name | - |PyTorch：通信域名称，MindSpore无此参数 |
