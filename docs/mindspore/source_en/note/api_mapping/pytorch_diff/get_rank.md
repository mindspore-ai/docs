# Function Differences with torch.distributed.get_rank

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/get_rank.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.distributed.get_rank

```python
torch.distributed.get_rank(group=None)
```

For more information, see [torch.distributed.get_rank](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.get_rank).

## mindspore.communication.get_rank

```python
mindspore.communication.get_rank(group=GlobalComm.WORLD_COMM_GROUP)
```

For more information, see [mindspore.communication.get_rank](https://mindspore.cn/docs/en/master/api_python/mindspore.communication.html#mindspore.communication.get_rank).

## Differences

PyTorch: The input of this interface is the communication group `group`.
The output is the `rank` of the process that calls this interface, and returns -1 if the process is not in `group`.

MindSpore: The input of this interface is the communication group `group`. The output is the `rank` of the process that calls this interface. Since `get_rank` should be called after `init`, the process that calls this interface should be in `group`. If `get_rank` is called before `init`, it will raise error.

| Class | Sub-class |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| Param  | Param 1 | group | group |No difference|
