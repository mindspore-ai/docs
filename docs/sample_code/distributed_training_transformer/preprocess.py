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
"""
Preprocess
"""
import pathlib

def wmt14_en_fr_preprocess(input_file, output_file):
    """Preprocess the source file and writes to the output_file"""
    input_file = input_file + "/newstest2014-fren-ref"
    output_file = output_file + "/wmt14"
    language = ['.en.sgm', '.fr.sgm']
    count = 0
    # en-fr
    with open(input_file + language[0], "r", encoding='utf-8') as english, \
            open(input_file + language[1], "r", encoding='utf-8') as french, \
            open(output_file + '.en_fr.txt', "a", encoding='utf-8') as enfr_f, \
            open(output_file + '.fr_en.txt', "a", encoding='utf-8') as fren_f:
        line_id = 0
        for en, fr in zip(english, french):
            line_id += 1
            if en[:7] == '<seg id':
                print("=" * 20, "\n", line_id, "\n", "=" * 20)
                en_start = en.find('>', 0)
                en_end = en.find('</seg>', 0)
                print(en[en_start + 1:en_end])
                en_ = en[en_start + 1:en_end]

                fr_start = fr.find('>', 0)
                fr_end = fr.find('</seg>', 0)
                print(fr[fr_start + 1:fr_end])
                fr_ = fr[fr_start + 1:fr_end]

                en_fr_str = en_ + "\t" + fr_ + "\n"
                enfr_f.write(en_fr_str)
                fr_en_str = fr_ + "\t" + en_ + "\n"
                fren_f.write(fr_en_str)
                count += 1

    print('write {} file finished!\n total count = {}'.format(output_file + '.en_fr.txt', count))
    print('write {} file finished!\n total count = {}'.format(output_file + '.fr_en.txt', count))

pathlib.Path('output').mkdir(exist_ok=True)
wmt14_en_fr_preprocess("test-full", 'output')
