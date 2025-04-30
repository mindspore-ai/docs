# Multi-device Model Weight Sharding

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_en/model_infer/ms_infer/weight_split.md)

After the model training is complete, the trained weights can be loaded for inference. The GPU memory required for inference is significantly lower than that required for training. Therefore, the model weights need to be sharded and loaded again.

## Weight Sharding

MindSpore uses the strategy file to manage distributed weights. After [model development](model_dev.md), the distribution attribute of weights in a frontend parallel network is determined by the basic module. Therefore, the weight sharding strategy information `sharded_state_dict` is first added to the distributed basic module.

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

The `sharded_state_dict` of the entire network may be generated based on the `sharded_state_dict` of the basic parallel module in the network. The network strategy information is obtained by calling `generate_state_dict` and saved as a strategy file by calling `save_strategy_file`.

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

After the parallel strategy file of the inference network is obtained, the training weight can be converted into the weight required for inference according to the method of executing distributed checkpoint transformation.

For details about the end-to-end weight sharding code project, see [Weight Sharding](https://gitee.com/mindspore/docs/blob/r2.6.0/docs/sample_code/infer_code/param_split.py).
