# Comparing the Function Differences with torch.distributed.new_group

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/create_group.md)

## torch.distributed.new_group

```python
torch.distributed.new_group(
    ranks=None,
    timeout=datetime.timedelta(0, 1800),
    backend=None
)
```

For more information, see [torch.distributed.new_group](https://pytorch.org/docs/1.8.1/distributed.html#torch.distributed.new_group).

## mindspore.communication.create_group

```python
mindspore.communication.create_group(group, rank_ids)
```

For more information, see [mindspore.communication.create_group](https://mindspore.cn/docs/en/r2.3/api_python/mindspore.communication.html#mindspore.communication.create_group).

## Differences

PyTorch: This interface passes in the rank list of the communication domain to be constructed, specifies the backend to create the specified communication domain, and returns the created communication domain.

MindSpore: This interface passes in the group name and the rank list of the communication domain to be constructed, creates a communication domain with the incoming group name as the key, and does not return any value.

| Class | Sub-class |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| Param | Param 1 | - | group | PyTorch does not have this param. MindSpore: the group name|
| | Param 2 | ranks | rank_ids | The functionalities are the same, but have different names|
| | Param 3 | timeout | - |PyTorch: the timeout value. MindSpore does not have this param, and should set the corresponding environment variables before calling this interface|
| | Param 4 | backend | - |PyTorch: the communication backend. MindSpore does not have this param, and should set the corresponding environment variables before calling this interface|