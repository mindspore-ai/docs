"""Reject sampling for math problem generation."""
# Copyright 2025 Huawei Technologies Co., Ltd
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
import argparse
import json
import re
from tqdm import tqdm
from math_verify import parse, verify


def get_last_boxed_content(text):
    # 查找所有 \boxed{} 的匹配项
    pattern = r'\\boxed\{(.*?)\}'
    matches = list(re.finditer(pattern, text))

    # 如果有匹配，返回最后一个的捕获组内容
    if matches:
        return matches[-1].group(1)  # group(1) 获取 {} 里的内容
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="Path to source json file.")
    parser.add_argument("--dst", type=str, help="Path to target mindrecrod file.")

    args = parser.parse_args()
    data = []
    with open(args.src, "r", encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))

    for data in tqdm(data):
        filtered_generations = [
            gen for gen in data['generations']
            if verify(parse(get_last_boxed_content(gen)), parse(data['answer']))
        ]
        if filtered_generations:
            data['generations'] = filtered_generations
            data['messages'] = [
                {"role": "user", "content": data['prompt']},
                {"content": filtered_generations[-1], "role": "assistant"}
                ]
            with open(args.dst, mode="a") as f:
                f.write(json.dumps(data) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
