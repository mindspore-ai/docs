# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" Transfer checkpoint for parallel model """

import os
import stat

from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.communication import get_group_size, get_rank, init
from mindspore.train.node_strategy_pb2 import ParallelStrategyMap as ckpt_strategy

from .model_dev import (ColumnParallelLinear, CommunicationHelper, ConfigHelper, ParallelTransformer, RowParallelLinear,
                        VocabParallelEmbedding)


def _update_sharded_state_dict(network: nn.Cell, dict_: dict):
    """Update the sharded state dict"""
    cells = network.name_cells()
    for _, subcell in cells.items():
        if subcell == network:
            continue
        if isinstance(subcell, (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)):
            dict_.update(subcell.sharded_state_dict())
        else:
            _update_sharded_state_dict(subcell, dict_)


def generate_state_dict(network):
    """Generate the sharded state dict for network"""
    state_dict = {
        "total_rank": get_group_size(),
        "stage_rank_size": get_group_size(),
        "stage": 0
    }
    model_state_dict = {}
    _update_sharded_state_dict(network=network, dict_=model_state_dict)
    state_dict['model'] = model_state_dict
    state_dict['optimizer'] = {}
    return state_dict


def save_strategy_file(state_dict, strategy_file_name):
    """ Save the strategy file according to the state_dict and strategy_file_name """
    stra = ckpt_strategy()

    # pylint: disable=W0612
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
        opt_weight_shard_step = item["opt_weight_shard_step"] if "opt_weight_shard_step" in item.keys() else 0
        opt_weight_shard_size = item["opt_weight_shard_size"] if "opt_weight_shard_size" in item.keys() else 0
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
            raise ValueError(
                f"For {name}, the shard{shard} requires {shard_mul} devices, "
                f"but the device number of this stage is {stage_rank_size}, "
                f"it can not be divisible by {shard_mul}"
            )
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

    if os.path.exists(strategy_file_name):
        os.chmod(strategy_file_name, stat.S_IWUSR)
    if "/" in strategy_file_name:
        real_path = os.path.abspath(strategy_file_name[: strategy_file_name.rfind("/")])
        os.makedirs(real_path, exist_ok=True)
    with open(strategy_file_name, "wb") as f:
        f.write(stra.SerializeToString())
        os.chmod(strategy_file_name, stat.S_IRUSR)


if __name__ == "__main__":
    COMMUN_HELPER = CommunicationHelper(group_name='tp', size=2)
    model_config = ConfigHelper(batch_size=64,
                                vocab_size=32000,
                                num_layers=4,
                                seq_length=2048,
                                hidden_size=1024,
                                ffn_hidden_size=4096,
                                dtype=mstype.float16,
                                num_heads=8,
                                has_bias=False)
    init()
    COMMUN_HELPER.create_tensor_model_parallel_group()
    parallel_transformer_2p = ParallelTransformer(config=model_config)

    strategy_info = generate_state_dict(network=parallel_transformer_2p)
    if get_rank() == 0:
        save_path = os.path.realpath('./output/strategy.ckpt')
        save_strategy_file(state_dict=strategy_info, strategy_file_name=save_path)
        print(f'Strategy file saved at {save_path}')
