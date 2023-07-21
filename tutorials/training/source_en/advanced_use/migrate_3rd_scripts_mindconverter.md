# Migrating From Third Party Frameworks With Tools

`Linux` `Ascend` `Model Development` `Beginner`

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_en/advanced_use/migrate_3rd_scripts_mindconverter.md)

## Overview

MindConverter is a migration tool to transform the model scripts from PyTorch to Mindspore. Users can migrate their PyTorch models to Mindspore rapidly with minor changes according to the conversion report.

## Installation

Mindconverter is a submodule in MindInsight. Please follow the [Guide](https://www.mindspore.cn/install/en) here to install MindInsight.

## Usage

MindConverter currently only provides command-line interface. Here is the manual page.

```bash
usage: mindconverter [-h] [--version] [--in_file IN_FILE]
                     [--model_file MODEL_FILE] [--shape SHAPE]
                     [--output OUTPUT] [--report REPORT]
                     [--project_path PROJECT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --version             show program version number and exit
  --in_file IN_FILE     Specify path for script file to use AST schema to do
                        script conversation.
  --model_file MODEL_FILE
                        PyTorch .pth model file path to use graph based schema
                        to do script generation. When `--in_file` and
                        `--model_file` are both provided, use AST schema as
                        default.
  --shape SHAPE         Optional, expected input tensor shape of
                        `--model_file`. It is required when use graph based
                        schema. Usage: --shape 3,244,244
  --output OUTPUT       Optional, specify path for converted script file
                        directory. Default output directory is `output` folder
                        in the current working directory.
  --report REPORT       Optional, specify report directory. Default is
                        converted script directory.
  --project_path PROJECT_PATH
                        Optional, PyTorch scripts project path. If PyTorch
                        project is not in PYTHONPATH, please assign
                        `--project_path` when use graph based schema. Usage:
                        --project_path ~/script_file/

```

**MindConverter provides two modes：**

1. **Abstract Syntax Tree (AST) based conversion**：Use the argument `--in_file` will enable the AST mode.
2. **Computational Graph basedconversion**：Use `--model_file` and `--shape` arguments will enable the Graph mode.

> The AST mode will be enabled, if both `--in_file` and `--model_file` are specified.

For the Graph mode, `--shape` is mandatory.

For the AST mode, `--shape` is ignored.

`--output` and `--report` is optional. MindConverter creates an `output` folder under the current working directory, and outputs generated scripts and conversion reports to it.  

Please note that your original PyTorch project is included in the module search path (PYTHONPATH). Use the python interpreter and test your module can be successfully loaded by `import` command. Use `--project_path` instead if your project is not in the PYTHONPATH to ensure MindConverter can load it.

> Assume the project is located at `/home/user/project/model_training`, users can use this command to add the project to `PYTHONPATH` : `export PYTHONPATH=/home/user/project/model_training:$PYTHONPATH`

> MindConverter needs the original PyTorch scripts because of the reverse serialization.

## Scenario

MindConverter provides two modes for different migration demands.

1. Keep original scripts' structures, including variables, functions, and libraries.
2. Keep extra modifications as few as possible, or no modifications are required after conversion.

The AST mode is recommended for the first demand. It parses and analyzes PyTorch scripts, then replace them with the MindSpore AST to generate codes. Theoretically, The AST mode supports any model script. However, the conversion may differ due to the coding style of original scripts.

For the second demand, the Graph mode is recommended. As the computational graph is a standard descriptive language, it is not affected by user's coding style. This mode may have more operators converted as long as these operators are supported by MindConverter.

Some typical image classification networks such as ResNet and VGG have been tested for the Graph mode. Note that:

> 1. Currently, the Graph mode does not support models with multiple inputs. Only models with a single input and single output are supported.
> 2. The Dropout operator will be lost after conversion because the inference mode is used to load the PyTorch model. Manually re-implement is necessary.
> 3. The Graph-based mode will be continuously developed and optimized with further updates.

## Example

### AST-Based Conversion

Assume the PyTorch script is located at `/home/user/model.py`, and outputs the transformed MindSpore script to `/home/user/output`, with the conversion report to `/home/user/output/report`. Use the following command:

```bash
mindconverter --in_file /home/user/model.py \
              --output /home/user/output \
              --report /home/user/output/report
```

In the conversion report, non-transformed code is listed as follows:

```text
line <row>:<col> [UnConvert] 'operator' didn't convert. ...
```

For non-transformed operators, the original code keeps. Please manually migrate them. [Click here](https://www.mindspore.cn/doc/note/en/r1.0/index.html#operator_api) for more information about operator mapping.

Here is an example of the conversion report:

```text
 [Start Convert]
 [Insert] 'import mindspore.ops.operations as P' is inserted to the converted file.
 line 1:0: [Convert] 'import torch' is converted to 'import mindspore'.
 ...
 line 157:23: [UnConvert] 'nn.AdaptiveAvgPool2d' didn't convert. Maybe could convert to mindspore.ops.operations.ReduceMean.
 ...
 [Convert Over]
```

For non-transformed operators, suggestions are provided in the report. For instance, MindConverter suggests that replace `torch.nn.AdaptiveAvgPool2d` with `mindspore.ops.operations.ReduceMean`.

### Graph-Based Conversion

Assume the PyTorch model (.pth file) is located at `/home/user/model.pth`, with input shape (3, 224, 224) and the original PyTorch script is at `/home/user/project/model_training`. Output the transformed MindSpore script to `/home/user/output`, with the conversion report to `/home/user/output/report`. Use the following command:

```bash
mindconverter --model_file /home/user/model.pth --shape 3,224,224 \
              --output /home/user/output \
              --report /home/user/output/report \
              --project_path /home/user/project/model_training
```

The Graph mode has the same conversion report as the AST mode. However, the line number and column number refer to the transformed scripts since no original scripts are used in the process.

In addition, input and output Tensor shape of unconverted operators shows explicitly (`input_shape` and `output_shape`) as comments in converted scripts to help further manual modifications. Here is an example of the `Reshape` operator (Not supported in current version):

```python
class Classifier(nn.Cell):

    def __init__(self):
        super(Classifier, self).__init__()
        ...
        self.reshape = onnx.Reshape(input_shape=(1, 1280, 1, 1),
                                    output_shape=(1, 1280))
        ...

    def construct(self, x):
        ...
        # Suppose input of `reshape` is x.
        reshape_output = self.reshape(x)
        ...

```

It is convenient to replace the operators according to the `input_shape` and `output_shape` parameters. The replacement is like this:

```python
import mindspore.ops as ops
...

class Classifier(nn.Cell):

    def __init__(self):
        super(Classifier, self).__init__()
        ...
        self.reshape = ops.Reshape(input_shape=(1, 1280, 1, 1),
                                 output_shape=(1, 1280))
        ...

    def construct(self, x):
        ...
        # Suppose input of `reshape` is x.
        reshape_output = self.reshape(x, (1, 1280))
        ...

```

> `--output` and `--report` are optional. MindConverter creates an `output` folder under the current working directory, and outputs generated scripts and conversion reports to it.

## Caution

1. PyTorch is not an explicitly stated dependency library in MindInsight. The Graph conversion requires the consistent PyTorch version as the model is trained. (MindConverter recommends PyTorch 1.4.0 or 1.6.0)
2. This script conversion tool relies on operators which supported by MindConverter and MindSpore. Unsupported operators may not successfully mapped to MindSpore operators. You can manually edit, or implement the mapping based on MindConverter, and contribute to our MindInsight repository. We appreciate your support for the MindSpore community.
