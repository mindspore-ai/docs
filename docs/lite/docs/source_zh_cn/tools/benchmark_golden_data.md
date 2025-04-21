# 标杆数据生成工具

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/tools/benchmark_golden_data.md)

## 概述

在对 MindSpore Lite 模型进行基准测试前，开发者可以按照约定格式生成标杆数据（Benchmark Data）。基于生成的标杆数据运用benchmark工具进行基准测试，能够对原始模型和转换后的模型精度进行定量分析。

mslite_gold标杆数据生成工具，可基于原始模型输入数据`input.npz`和推理得到的输出数据`output.npz`生成标杆数据，本文介绍该工具的用法。

## mslite_gold工具使用说明

开发者首先需要将原始模型输入数据和推理得到的输出数据，通过`numpy`中的`savez()`命令保存为`.npz`格式的文件，再通过运行[mslite_gold.py](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/sample_code/golden/mslite_gold.py) 分别将输入数据`input.npz`和输出数据`output.npz`转化为`.bin`和`.out`的二进制格式文件。

### 环境要求

python依赖库

- `numpy` >= 2.0.2
- `onnx` >= 1.17.0
- `onnxruntime` >= 1.19.2
- `ast` >= 1.6.3
- `collection` >= 0.1.6

### 使用示例

```bash
python mslite_gold.py --inputFile "/path/to/input.npz" --outputFile "/path/to/output.npz" --savePath "/path/to/save_data"
```

执行命令后，会在`/path/to/save_data/`目录中生成如下文件：

- `input.bin`：输入数据，对每个输入Tensor的数据进行`flatten`一维化，单独保存到一个二进制文件`input.bin`，不含名称、dtype、shape等元信息。
- `output.out`：输出数据，所有输出内容保存到一个文本文件，格式如下：

每个输出Tensor占用2行：

```bash
name dim shape1 [shape2 ...]
data0 data1 ...
```

- 第一行：记录输出Tensor的元信息，依次为：名称、维度、形状
- 第二行：记录输出Tensor的数据。

示例：

```bash
out_0 2 2 3
1.0 2.0 3.0 4.0 5.0 6.0
```

表示一个名称为`out_0`，维度是2维，形状是[2, 3]，内容是`[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]`的Tensor。

下面以ONNX模型生成标杆数据为例，详细介绍工具的用法。

1. 随机生成输入，进行模型推理，再将输入输出保存为`.npz`格式。

   [onnx_demo.py示例代码](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/sample_code/golden/onnx_demo.py) 支持基于ONNX模型随机生成数据，或手动输入数据，并执行推理获得输出数据。手动输入数据时，可以通过参数`--inDataFile`来确定输入数据路径。如果模型是动态shape类型，需通过参数`--inputShape`来确定输入尺寸。参数`--inDataFile`和`--inputShape`非必须，用户可以根据自身的使用场景来自由选用。下面是示例代码的基础使用示例：

   ```bash
    python onnx_demo.py --modelFile "/path/to/model_example.onnx" --savePath "/path/to/data_example"
   ```

   `/path/to/data_example/`目录中会生成`input.npz`和`output.npz`文件。

2. 将输入输出的`npz`文件转化成标杆数据文件。

   ```bash
   python mslite_gold.py --inputFile "/path/to/data_example/input.npz" --outputFile "/path/to/data_example/output.npz" --savePath "/path/to/save_data"
   ```

   最后，在`/path/to/save_data`目录中会生成上述的标杆数据文件。

### mslite_gold工具代码细节

以下展示标杆数据生成的代码细节，先转换input，再转换output。

```python
import os
import numpy as np
from collections import OrderedDict

def save_bin(args):
    try:
        input_dict = np.load(args.inputFile)
        print(f"Loaded inputs from {args.inputFile}: {list(input_dict.keys())}")
    except Exception as e:
        print(f"Error loading inputs: {e}")
        return
    i = 0
    input_dict=OrderedDict(input_dict)
    for key , input_data in input_dict.items():
        print(f"input {key}: shape={input_data.shape}, dtype={input_data.dtype}")
        if np.issubdtype(input_data.dtype, np.integer):
            input_data.astype(np.int32).flatten().tofile(os.path.join(args.savePath, f"input.bin{i}"))
        else:
            input_data.flatten().tofile(os.path.join(args.savePath, f"input.bin{i}"))
        i = i + 1

    try:
        output_dict = np.load(args.outputFile)
        print(f"Loaded outputs from {args.outputFile}: {list(output_dict.keys())}")
    except Exception as e:
        print(f"Error loading outputs: {e}")
        return

    opened = 0
    output_dict = OrderedDict(output_dict)
    output_file = os.path.join(args.savePath, 'model.out')
    for i , output_data in output_dict.items():
        print(f"output {i}: shape={output_data.shape}, dtype={output_data.dtype}")
        mode = 'w' if opened == 0 else 'a'
        if str(output_data.shape) == "[]":
            output_shape = [1]
        else:
            output_shape = output_data.shape

        with open(output_file, mode) as text_file:
            opened = 1
            if len(output_shape) == 0:
                output_shape.append(len(output_data))

            text_file.write(f"{i} {len(output_data.shape)} ")
            text_file.write(" ".join([str(s) for s in output_data.shape]))
            text_file.write('\n')
            print(f"result shape: {len(output_data.flatten())}")

            for k in output_data.flatten():
                text_file.write(f"{k} ")
            text_file.write('\n')
```
