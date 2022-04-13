# Function Differences with torch.distributed.init_process_group

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/init.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [torch.distributed.init_process_group](https://pytorch.org/docs/1.5.0/distributed.html#torch.distributed.init_process_group).

## mindspore.communication.init

```python
mindspore.communication.init(backend_name=None)
```

For more information, see [mindspore.communication.init](https://mindspore.cn/docs/en/master/api_python/mindspore.communication.html#mindspore.communication.init).

## Differences

PyTorch: This interface supports three kinds of collective communications: MPI, Gloo, and NCCL. It initializes `backend` and also provides configuration, such as `world_size`, `rank`, `timeout`, etc.

MindSpore：This interface currently supports only two kinds of collective communication: HCCL and NCCL. The configuration of `world_size`, `rank` and `timeout` is not set in this interface. The corresponding environment variable needs to be set before calling this interface.

