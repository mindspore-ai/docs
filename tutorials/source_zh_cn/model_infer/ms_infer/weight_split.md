# 多卡模型权重切分

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_zh_cn/model_infer/ms_infer/weight_split.md)

模型在完成训练后，可以加载训练好的权重进行推理，而推理所需的显存较训练时显著降低，因此需要对模型权重进行重新切分和加载。

## 权重切分

MindSpore依靠strategy文件管理分布式权重。在[模型开发](model_dev.md)后，对于前端并行的网络而言，其权重的分布式属性是由基础模块决定的。因此，首先为分布式基础模块添加权重切分策略信息`sharded_state_dict`。

```python
class ColumnParallelLinear(nn.Cell):
...
    def sharded_state_dict(self):
        w_shard = (self.tensor_parallel_group_size, 1) if self.transpose_b else (1, self.tensor_parallel_group_size)
        state_dict = {}
        if not self.skip_weight_param_allocation:
            state_dict[self.weight.name] = {'shape': self.weight.shape,
                                            'shard': w_shard}
        if self.has_bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (self.tensor_parallel_group_size,)}
        return state_dict
```

```python
class RowParallelLinear(nn.Cell):
...
    def sharded_state_dict(self):
        w_shard = (1, self.tensor_parallel_group_size) if self.transpose_b else (self.tensor_parallel_group_size, 1)
        state_dict = {}
        state_dict[self.weight.name] = {'shape': self.weight.shape,
                                        'shard': w_shard}
        if self.has_bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (1,)}
        return state_dict
```

```python
class VocabParallelEmbedding(nn.Cell):
...
    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        w_shard = (self.tensor_model_parallel_size, 1)
        state_dict = {}
        state_dict[self.weight.name] = {'shape': self.weight.shape,
                                        'shard': w_shard}
        return state_dict
```

基于网络中基础并行模块的`sharded_state_dict`，可以生成整个网络的`sharded_state_dict`。通过调用`generate_state_dict`将得到网络的策略信息，并通过`save_strategy_file`保存为策略文件。

```python
def _update_sharded_state_dict(network: nn.Cell, dict_: dict):
    cells = network.name_cells()
    for _, subcell in cells.items():
        if subcell == network:
            continue
        if isinstance(subcell, (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)):
            dict_.update(subcell.sharded_state_dict())
        else:
            _update_sharded_state_dict(subcell, dict_)

def generate_state_dict(network):
    state_dict = {
        "total_rank": get_group_size(),
        "stage_rank_size": get_group_size(),
        "stage": 0
    }
    model_state_dict = {}
    _update_sharded_state_dict(network=network, dict_=model_state_dict)
    state_dict['model'] = model_state_dict
    return state_dict

def save_strategy_file(state_dict, strategy_file_name):
    import os
    import stat
    from mindspore.train.node_strategy_pb2 import ParallelStrategyMap as ckpt_strategy
    stra = ckpt_strategy()

    total_rank = state_dict["total_rank"]
    stage_rank_size = state_dict["stage_rank_size"]
    stage = state_dict["stage"]
    model_param = state_dict["model"]
    optimizer_param = state_dict["optimizer"]
    stra.current_stage = 0
    model_param.update(optimizer_param)
    for name, item in model_param.items():
        if "shard" not in item or "shape" not in item:
            continue
        opt_weight_shard_step = item["opt_weight_shard_step"] \
            if "opt_weight_shard_step" in item.keys() else 0
        opt_weight_shard_size = item["opt_weight_shard_size"] \
            if "opt_weight_shard_size" in item.keys() else 0
        strategy_item = stra.parallel_strategy_item.add()
        strategy_item.node_name = name
        parallel_strategys = strategy_item.parallel_strategys
        parallel_strategys.stage = stage
        shard = item["shard"]
        shape = item["shape"]
        parallel_strategy = parallel_strategys.parallel_strategy.add()
        shard_mul = 1
        for ele in shard:
            parallel_strategy.dim.append(ele)
            shard_mul = shard_mul * ele
        layout_item = stra.parallel_layout_item.add()
        layout_item.param_name = name
        parallel_layouts = layout_item.parallel_layouts
        parallel_layouts.field = 0
        parallel_layouts.opt_weight_shard_step = opt_weight_shard_step
        parallel_layouts.opt_weight_shard_size = opt_weight_shard_size
        dev_matrix = parallel_layouts.dev_matrix.add()
        repeat_calc_num = 1
        if stage_rank_size == shard_mul:
            repeat_calc_num = 1
        elif stage_rank_size % shard_mul == 0:
            repeat_calc_num = stage_rank_size // shard_mul
        else:
            raise ValueError(f"For {name}, the shard{shard} requires {shard_mul} devices, "
                             f"but the device number of this stage is {stage_rank_size}, "
                             f"it can not be divisible by {shard_mul}")
        if repeat_calc_num != 1:
            dev_matrix.dim.append(repeat_calc_num)
        for ele in shard:
            dev_matrix.dim.append(ele)
        tensor_map = parallel_layouts.tensor_map.add()
        shape_len = len(shape)
        index = shape_len - 1
        for _ in range(shape_len):
            tensor_map.dim.append(index)
            index = index - 1
        param_split_shape = parallel_layouts.param_split_shape.add()
        for ele in shape:
            param_split_shape.dim.append(ele)

    try:
        if os.path.exists(strategy_file_name):
            os.chmod(strategy_file_name, stat.S_IWUSR)
        if "/" in strategy_file_name:
            real_path = os.path.abspath(strategy_file_name[:strategy_file_name.rfind("/")])
            os.makedirs(real_path, exist_ok=True)
        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(strategy_file_name, flags_, 0o750), 'wb') as f:
            f.write(stra.SerializeToString())
            os.chmod(strategy_file_name, stat.S_IRUSR)

    except BaseException as e:
        logger.critical(f"Failed to save the checkpoint file {strategy_file_name}. Maybe don't have "
                        "the permission to write files, or the disk space is insufficient and so on.")
        raise e
```

得到推理网络的并行策略文件后，可以根据执行分布式checkpoint转换方法，将训练权重转换为推理所需权重。

具体端到端的权重切分代码工程可以参考[权重切分](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/sample_code/infer_code/param_split.py)。
