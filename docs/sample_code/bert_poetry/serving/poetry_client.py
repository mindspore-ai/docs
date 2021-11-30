# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Serving client."""
import time
import re
from mindspore_serving.client import Client

def predict(inputs, generate_type):
    """Client for servable bert poetry"""
    client = Client("127.0.0.1:5500", "bert", "predict")
    instance = {"inputs": inputs, "types": generate_type}
    result = client.infer(instance)
    return result[0]['poetry']

while True:
    print("\n*************************************")
    types = input("选择模式：0-随机生成，1-续写，2-藏头诗\n")
    try:
        types = int(types)
    except ValueError:
        continue
    if types not in [0, 1, 2]:
        continue
    if types == 1:
        s = input("输入首句诗\n")
    elif types == 2:
        s = input("输入藏头诗\n")
    else:
        s = ''
    start_time = time.time()
    predictions = predict(s, types)
    end_to_end_delay = (time.time() - start_time) * 1000
    a = re.findall(r'[\u4e00-\u9fa5]*[\uff0c\u3002]', predictions)
    print("\n")
    for poem in a:
        print(poem)
    print("\ncost time: {:.1f} ms".format(end_to_end_delay))
