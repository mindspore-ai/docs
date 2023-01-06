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

For more information, see [torch.distributed.init_process_group](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.init_process_group).

## mindspore.communication.init

```python
mindspore.communication.init(backend_name=None)
```

For more information, see [mindspore.communication.init](https://mindspore.cn/docs/en/master/api_python/mindspore.communication.html#mindspore.communication.init).

## Differences

PyTorch: This interface supports three kinds of collective communications: MPI, Gloo, and NCCL. It initializes `backend` and also provides configuration, such as `world_size`, `rank`, `timeout`, etc.

MindSpore：This interface currently supports three kinds of collective communication: HCCL, NCCL, and MCCL. The configuration of `world_size`, `rank` and `timeout` is not set in this interface. The corresponding environment variables need to be set before calling this interface.

| Class | Sub-class |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | backend | backend_name | The functionalities are the same, but have different names and support different communication backend|
| | Parameter 2 | init_method | - | PyTorch: the initialization method. MindSpore does not have this param|
| | Parameter 3 | timeout | - |PyTorch: the timeout value. MindSpore does not have this parameter, and should set the corresponding environment variables before calling this interface|
| | Parameter 4 | world_size | - |PyTorch: the world size of communication group. MindSpore does not have this parameter, and should set the corresponding environment variables before calling this interface|
| | Parameter 5 | rank | - |PyTorch: the current rank. MindSpore does not have this parameter, and should set the corresponding environment variables before calling this interface|
| | Parameter 6 | store | - |PyTorch: the flag of storing key/value. MindSpore does not have this parameter |
| | Parameter 7 | group_name | - |PyTorch: the group name. MindSpore does not have this parameter |