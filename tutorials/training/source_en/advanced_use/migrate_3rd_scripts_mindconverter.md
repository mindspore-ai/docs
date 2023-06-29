# Migrating From Third Party Frameworks With Tools

`Linux` `Ascend` `Model Development` `Beginner`

<a href="https://gitee.com/mindspore/docs/blob/r1.2/tutorials/training/source_en/advanced_use/migrate_3rd_scripts_mindconverter.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

MindConverter is a migration tool to transform the model scripts and weights from PyTorch(ONNX) and TensorFlow(PB) to MindSpore. Users can migrate rapidly with minor changes according to the conversion report.

## Installation

Mindconverter is a submodule in MindInsight. Please follow the [Guide](https://gitee.com/mindspore/mindinsight/blob/r1.2/README.md#) here to install MindInsight.

## Usage

MindConverter currently only provides command-line interface. Here is the manual page.

```bash
usage: mindconverter [-h] [--version] [--in_file IN_FILE]
                     [--model_file MODEL_FILE] [--shape SHAPE [SHAPE ...]]
                     [--input_nodes INPUT_NODES [INPUT_NODES ...]]
                     [--output_nodes OUTPUT_NODES [OUTPUT_NODES ...]]
                     [--output OUTPUT] [--report REPORT]

optional arguments:
  -h, --help            show this help message and exit
  --version             show program version number and exit
  --in_file IN_FILE     Specify path for script file to use AST schema to do
                        script conversation.
  --model_file MODEL_FILE
                        Tensorflow(.pb) or ONNX(.onnx) model file path is
                        expected to do script generation based on graph
                        schema. When `--in_file` and `--model_file` are both
                        provided, use AST schema as default.
  --shape SHAPE [SHAPE ...]
                        Optional, expected input tensor shape of
                        `--model_file`. It is required when use graph based
                        schema. Both order and number should be consistent
                        with `--input_nodes`. Usage: --shape 1,512 1,512
  --input_nodes INPUT_NODES [INPUT_NODES ...]
                        Optional, input node(s) name of `--model_file`. It is
                        required when use graph based schema. Both order and
                        number should be consistent with `--shape`. Usage:
                        --input_nodes input_1:0 input_2:0
  --output_nodes OUTPUT_NODES [OUTPUT_NODES ...]
                        Optional, output node(s) name of `--model_file`. It is
                        required when use graph based schema. Usage:
                        --output_nodes output_1:0 output_2:0
  --output OUTPUT       Optional, specify path for converted script file
                        directory. Default output directory is `output` folder
                        in the current working directory.
  --report REPORT       Optional, specify report directory. Default is
                        converted script directory.
```

### PyTorch Model Scripts Migration

**MindConverter only provides Abstract Syntax Tree (AST) based conversion for PyTorch**: Use the argument `--in_file` will enable the AST mode.

> The AST mode will be enabled, if both `--in_file` and `--model_file` are specified.

`--output` and `--report` is optional. MindConverter creates an `output` folder under the current working directory, and outputs generated scripts and conversion reports to it.  

> If the user want to migrate PyTorch model script using graph based MindConverter, it is recommended to export PyTorch model to ONNX, and then use ONNX file to migrate model script. For details, see [PyTorch instructions](https://pytorch.org/docs/stable/onnx.html).

### TensorFlow Model Scripts Migration

**MindConverter provides computational graph based conversion for TensorFlow**: Transformation will be done given `--model_file`, `--shape`, `--input_nodes` and `--output_nodes`.

> AST mode is not supported for TensorFlow, only computational graph based mode is available.

`--output` and `--report` is optional. MindConverter creates an `output` folder under the current working directory, and outputs generated scripts to it.  

### ONNX Model File Migration

**MindConverter provides computational graph based conversion for ONNX**: Transformation will be done given `--model_file`, `--shape`, `--input_nodes` and `--output_nodes`.

> AST mode is not supported for ONNX, only computational graph based mode is available.

`--output` and `--report` is optional. MindConverter creates an `output` folder under the current working directory, and outputs generated scripts to it.  

## Scenario

MindConverter provides two modes for different migration demands.

1. Keep original scripts' structures, including variables, functions, and libraries.
2. Keep extra modifications as few as possible, or no modifications are required after conversion.

The AST mode is recommended for the first demand (AST mode is only supported for PyTorch). It parses and analyzes PyTorch scripts, then replace them with the MindSpore AST to generate codes. Theoretically, The AST mode supports any model script. However, the conversion may differ due to the coding style of original scripts.

For the second demand, the Graph mode is recommended. As the computational graph is a standard descriptive language, it is not affected by user's coding style. This mode may have more operators converted as long as these operators are supported by MindConverter.

Some typical networks in computer vision field have been tested for the Graph mode. Note that:

> 1. The Dropout operator will be lost after conversion because the inference mode is used to load the ONNX or TensorFlow model. Manually re-implement is necessary.
> 2. The Graph-based mode will be continuously developed and optimized with further updates.

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

For non-transformed operators, the original code keeps. Please manually migrate them. [Click here](https://www.mindspore.cn/doc/note/en/r1.2/index.html#operator_api) for more information about operator mapping.

Here is an example of the conversion report:

```text
 [Start Convert]
 [Insert] 'import mindspore.ops as ops' is inserted to the converted file.
 line 1:0: [Convert] 'import torch' is converted to 'import mindspore'.
 ...
 line 157:23: [UnConvert] 'nn.AdaptiveAvgPool2d' didn't convert. Maybe could convert to mindspore.ops.operations.ReduceMean.
 ...
 [Convert Over]
```

For non-transformed operators, suggestions are provided in the report. For instance, MindConverter suggests that replace `torch.nn.AdaptiveAvgPool2d` with `mindspore.ops.operations.ReduceMean`.

### Graph-Based Conversion

#### TensorFlow Model Scripts Conversion

To use TensorFlow model script migration, you need to export TensorFlow model to Pb format(frozen graph) first, and obtain the model input node and output node name. See [Tutorial of exporting TensorFlow Pb model](https://gitee.com/mindspore/mindinsight/blob/r1.2/mindinsight/mindconverter/docs/tensorflow_model_exporting.md#) for details.

Suppose the model is saved to `/home/user/xxx/frozen_model.pb`, corresponding input node name is `input_1:0`, output node name is `predictions/Softmax:0`, the input shape of model is `1,224,224,3`. Output the transformed MindSpore script to `/home/user/output`, with the conversion report to `/home/user/output/report`. Use the following command:

```bash
mindconverter --model_file /home/user/xxx/frozen_model.pb --shape 1,224,224,3 \
              --input_nodes input_1:0 \
              --output_nodes predictions/Softmax:0 \
              --output /home/user/output \
              --report /home/user/output/report
```

After executing the command, MindSpore script, MindSpore weight file, weight map file, and report file can be found in corresponding directory.

The format of conversion report generated by script generation scheme based on graph structure is the same as that of AST scheme. However, since the graph based scheme is a generative method, the original tensorflow script is not referenced in the conversion process. Therefore, the code line and column numbers involved in the generated conversion report refer to the generated script.

In addition, input and output Tensor shape of unconverted operators shows explicitly (`input_shape` and `output_shape`) as comments in converted scripts to help further manual modifications. Here is an example of the `Reshape` operator (already supported after R1.0 version):

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

Weight information in MindSpore (`converted_weight`) and that in source framework(`source_weight`) are saved in weight map separately.

Here is an example of the weight map:

```json
{
    "resnet50": [
        {
            "converted_weight": {
                "name": "conv2d_0.weight",
                "shape": [
                    64,
                    3,
                    7,
                    7
                ],
                "data_type": "Float32"
            },
            "source_weight": {
                "name": "conv1.weight",
                "shape": [
                    64,
                    3,
                    7,
                    7
                ],
                "data_type": "float32"
            }
        }
    ]
}
```

#### ONNX Model File Conversion

To use ONNX model File migration, you need to obtain the model input node and output node names. To get input node and output node names, [Netron](https://github.com/lutzroeder/netron) is recommended.

Suppose the model is saved to `/home/user/xxx/model.onnx`, the corresponding input node name is `input_1:0`, the output node name is `predictions/Softmax:0`, the input shape of model is `1,3,224,224`, the following command can be used to generate the script:

```bash
mindconverter --model_file /home/user/xxx/model.onnx --shape 1,3,224,224 \
              --input_nodes input_1:0 \
              --output_nodes predictions/Softmax:0 \
              --output /home/user/output \
              --report /home/user/output/report
```

After executed, MindSpore script, MindSpore weight file, weight map file, and report file can be found in corresponding directory.

The format of conversion report generated by script generation scheme based on graph structure is the same as that of AST scheme. However, since the graph based scheme is a generative method, the original onnx file is not referenced in the conversion process. Therefore, the code line and column numbers involved in the generated conversion report refer to the generated script.

The example of weight map refers to that in **TensorFlow Model Scripts Conversion** section.

## MindConverter Error Code Definition

Error code defined in MindConverter, please refer to [LINK](https://gitee.com/mindspore/mindinsight/blob/r1.2/mindinsight/mindconverter/docs/error_code_definition.md# ).

## Model List Supported by MindConverter

[List of supported models (Models in below table have been tested based on PyTorch 1.5.0 and TensorFlow 1.15.0, X86 Ubuntu released version)](https://gitee.com/mindspore/mindinsight/blob/r1.2/mindinsight/mindconverter/docs/supported_model_list.md# ).

## Caution

1. TensorFlow is not a dependency library explicitly declared by MindInsight. If the user want to use graph based MindConverter, please install TensorFlow(MindConverter recommends TensorFlow 1.15.x).
2. ONNX(>=1.8.0), ONNXRUNTIME(>=1.5.2), ONNXOPTIMIZER(>=0.1.2) are not explicitly stated dependency libraries in MindInsight, if the user want to use graph based MindConverter, above three-party libraries must be installed. If the user want to migrate TensorFlow model to MindSpore, TF2ONNX(>=1.7.1) must be installed additionally.
3. This script conversion tool relies on operators which supported by MindConverter and MindSpore. Unsupported operators may not be successfully mapped to MindSpore operators. You can manually edit, or implement the mapping based on MindConverter, and contribute to our MindInsight repository. We appreciate your support for the MindSpore community.
4. MindConverter converts dynamic input shape to constant one based on `--shape` while using grpah based scheme, as a result, it is required that inputs' shape used to retrain or inference in MindSpore are the same as that used to convert using MindConverter. If the input shape has changed, please re-running MindConverter with new `--shape` or fixing shape related parameters in the old script.
5. MindSpore script, MindSpore checkpoint file and weight map file are saved in the same file folder path.
6. The security and consistency of the model file should be guaranteed by the user.

