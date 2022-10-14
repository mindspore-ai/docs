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
Transform distributed checkpoint by directory
"""
import argparse
import mindspore as ms

def main():
    # Transform checkpoint directory
    parser = argparse.ArgumentParser(description="Transform checkpoint dir")
    parser.add_argument('--src_strategy_dir',
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
    ms.merge_pipeline_strategys(args_opt.src_strategy_dir, "./merged_pipeline_strategy.ckpt")
    ms.transform_checkpoints(args_opt.src_checkpoints_dir, args_opt.dst_checkpoints_dir,
                             "transformed", "./merged_pipeline_strategy.ckpt", args_opt.dst_strategy_file)

if __name__ == "__main__":
    main()
