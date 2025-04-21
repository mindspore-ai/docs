# Benchmark Data Generation Tool

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/tools/benchmark_golden_data.md)

## Overview

Before benchmarking MindSpore Lite models, developers can generate benchmark data in an agreed format. Benchmarking based on the generated benchmark data using the benchmark tool enables quantitative analysis of the accuracy of the original and converted models.

This paper describes the usage of the mslite_gold benchmarking data generation tool, which generates benchmarking data based on the original model input data `input.npz` and the output data `output.npz` obtained by inference.

## Instructions for Using the mslite_gold Tool

The developers first need to save the original model input data and the output data obtained from inference to a file in `.npz` format via the `savez()` command in `numpy`, and then convert the input data `input.npz` and the output data `output.npz` into `.bin` and `.out` files in binary format respectively by running [mslite_gold.py](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/sample_code/golden/mslite_gold.py).

### Environmental Requirements

Python dependency libraries

- `numpy` >= 2.0.2
- `onnx` >= 1.17.0
- `onnxruntime` >= 1.19.2
- `ast` >= 1.6.3
- `collection` >= 0.1.6

### Usage Example

```bash
python mslite_gold.py --inputFile "/path/to/input.npz" --outputFile "/path/to/output.npz" --savePath "/path/to/save_data"
```

After executing the command, the following file is generated in the `/path/to/save_data/` directory:

- `input.bin`: Input data, `flatten` the data of each input Tensor one-dimensionally and save it separately to a binary file `input.bin` without meta-information such as name, dtype, and shape.
- `output.out`: Output data, all output is saved to a text file with the following format:

Each output Tensor occupies 2 lines:

```bash
name dim shape1 [shape2 ...]
data0 data1 ...
```

- First line: record the meta information of the output Tensor, in order: name, dimension, shape.
- Second line: record the data from the output Tensor.

For example:

```bash
out_0 2 2 3
1.0 2.0 3.0 4.0 5.0 6.0
```

which represents a Tensor with name `out_0`, 2-dimension, shape [2, 3] and content `[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]`.

The following is an example of generating benchmark data from an ONNX model to introduce the usage of the tool in detail.

1. Randomly generate inputs, perform model inference, and then save the inputs and outputs in `.npz` format.

   [onnx_demo.py sample code](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/sample_code/golden/onnx_demo.py) supports random data generation based on ONNX models, or manually inputting data and performing inference to obtain output data. When inputting data manually, the input data path can be determined by the parameter `--inDataFile`. If the model is of dynamic shape type, the input size should be determined by the parameter `--inputShape`. Parameters `--inDataFile` and `--inputShape` are not required, and users can choose freely according to their own use of the scenario. The following is a basic example of the use of sample code:

   ```bash
    python onnx_demo.py --modelFile "/path/to/model_example.onnx" --savePath "/path/to/data_example"
   ```

   The `input.npz` and `output.npz` files are generated in the `/path/to/data_example/` directory.

2. Converts the input and output `npz` files into a benchmark data file.

   ```bash
   python mslite_gold.py --inputFile "/path/to/data_example/input.npz" --outputFile "/path/to/data_example/output.npz" --savePath "/path/to/save_data"
   ```

   Finally, the benchmark data file described above will be generated in the `/path/to/save_data` directory.

### mslite_gold Tool Code Details

The following shows the details of the code for the generation of the benchmarking data, first converting the input and then converting the output.

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
