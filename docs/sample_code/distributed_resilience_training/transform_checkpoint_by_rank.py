# Copyright 2022 Huawei Technologies Co., Ltd

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
"""
Transform distributed checkpoint by rank
"""
import os
import argparse
import mindspore as ms


def main():
    # Transform checkpoint directory
    parser = argparse.ArgumentParser(description="Transform checkpoint dir")
    parser.add_argument('--transform_rank',
                        type=int,
                        default=0,
                        help="The rank to transform, default is 0.")
    parser.add_argument('--src_strategy_file',
                        type=str,
                        default=None,
                        help="The source strategy file, default is None.")
    parser.add_argument("--dst_strategy_file",
                        type=str,
                        default=None,
                        help="The destination strategy file, default is None.")
    parser.add_argument("--src_checkpoints_dir",
                        type=str,
                        default="./src_ckpt",
                        help="The source checkpoint directory.")
    parser.add_argument("--dst_checkpoints_dir",
                        type=str,
                        default="./dst_ckpt",
                        help="The destination checkpoint directory.")
    args_opt = parser.parse_args()
    rank_list = ms.rank_list_for_transform(args_opt.transform_rank,
                                           args_opt.src_strategy_file, args_opt.dst_strategy_file)
    checkpoint_file_map = {}
    for rank_id in rank_list:
        checkpoint_file_map[rank_id] = os.path.join(args_opt.src_checkpoints_dir,
                                                    "rank_{}".format(rank_id), "src_checkpoint-2_73.ckpt")
    save_checkpoint_path = os.path.join(args_opt.dst_checkpoints_dir, "rank_{}".format(args_opt.transform_rank),
                                        "transformed{}.ckpt".format(args_opt.transform_rank))
    ms.transform_checkpoint_by_rank(args_opt.transform_rank, checkpoint_file_map, save_checkpoint_path,
                                    args_opt.src_strategy_file, args_opt.dst_strategy_file)
if __name__ == "__main__":
    main()
