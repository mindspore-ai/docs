# Copyright 2022 Huawei Technologies Co., Ltd
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

"""This is a tool to get the path of dumped tensor files."""
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dump_config', type=str, default='./async_dump.json', help='the path of dump config file')
args = parser.parse_args()

with open(args.dump_config) as json_file:
    data = json.load(json_file)

settings = data["common_dump_settings"]
path = settings["path"]
net = settings["net_name"]
iteration = settings["iteration"]
root_graph_id = "0"
tensor_path = os.path.join(path, "rank_0", net, root_graph_id, iteration)
print(tensor_path)
