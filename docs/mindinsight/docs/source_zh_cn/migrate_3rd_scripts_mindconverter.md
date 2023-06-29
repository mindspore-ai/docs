# 使用工具迁移模型定义脚本

`Linux` `Ascend` `模型开发` `初级`

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindinsight/docs/source_zh_cn/migrate_3rd_scripts_mindconverter.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

## 概述

MindConverter是一款用于将PyTorch（ONNX）、TensorFlow（PB）模型转换到MindSpore模型定义脚本以及权重文件的工具。结合转换报告的信息，用户只需对转换后的脚本进行微小的改动，即可实现快速迁移。

## 安装

此工具为MindInsight的子模块，安装MindInsight后，即可使用MindConverter，MindInsight安装请参考该[安装文档](https://gitee.com/mindspore/mindinsight/blob/r1.3/README_CN.md#)。

除安装MindInsight之外，还需要安装下列依赖库：

1. TensorFlow不作为MindInsight明确声明的依赖库。若想使用基于图结构的脚本生成工具，需要用户手动安装TensorFlow（MindConverter推荐使用TensorFlow 1.15.x版本）。
2. ONNX（>=1.8.0）、ONNXRUNTIME（>=1.5.2）、ONNXOPTIMIZER（>=0.1.2）不作为MindInsight明确声明的依赖库，若想使用基于图结构的脚本生成工具，必须安装上述三方库。若想使用TensorFlow（MindConverter推荐使用TensorFlow 1.15.x版本）模型脚本迁移需要额外安装TF2ONNX（>=1.7.1）。

## 用法

MindConverter提供命令行（Command-line interface, CLI）的使用方式，命令如下。

```text
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
                        Expected input tensor shape of `--model_file`. It is
                        required when use graph based schema. Both order and
                        number should be consistent with `--input_nodes`.
                        Given that (1,128) and (1,512) are shapes of input_1
                        and input_2 separately. Usage: --shape 1,128 1,512
  --input_nodes INPUT_NODES [INPUT_NODES ...]
                        Input node(s) name of `--model_file`. It is required
                        when use graph based schema. Both order and number
                        should be consistent with `--shape`. Given that both
                        input_1 and input_2 are inputs of model. Usage:
                        --input_nodes input_1 input_2
  --output_nodes OUTPUT_NODES [OUTPUT_NODES ...]
                        Output node(s) name of `--model_file`. It is required
                        when use graph based schema. Given that both output_1
                        and output_2 are outputs of model. Usage:
                        --output_nodes output_1 output_2
  --output OUTPUT       Optional, specify path for converted script file
                        directory. Default output directory is `output` folder
                        in the current working directory.
  --report REPORT       Optional, specify report directory. Default is
                        converted script directory.
```

### PyTorch模型脚本迁移

**MindConverter仅提供基于抽象语法树（Abstract syntax tree, AST）的PyTorch脚本迁移**：指定`--in_file`的值，将使用基于AST的脚本转换方案；

> 若同时指定了`--in_file`，`--model_file`将默认使用AST方案进行脚本迁移。

其中，`--output`与`--report`参数可省略。若省略，MindConverter将在当前工作目录（Working directory）下自动创建`output`目录，将生成的脚本、转换报告输出至该目录。

> 若需要使用MindConverter计算图方案进行PyTorch模型脚本迁移，建议将PyTorch模型转换为ONNX，再使用ONNX文件进行模型脚本迁移，详情见[PyTorch使用说明](https://pytorch.org/docs/stable/onnx.html)。

### TensorFlow模型脚本迁移

**MindConverter提供基于图结构的脚本生成方案**：指定`--model_file`、`--shape`、`--input_nodes`、`--output_nodes`进行脚本迁移。

> AST方案不支持TensorFlow模型脚本迁移，TensorFlow脚本迁移仅支持基于图结构的方案。

若省略`--output`与`--report`参数，MindConverter将在当前工作目录（Working directory）下自动创建`output`目录，将生成的脚本、转换报告、权重文件、权重映射表输出至该目录。

### ONNX模型文件迁移

**MindConverter提供基于图结构的脚本生成方案**：指定`--model_file`、`--shape`、`--input_nodes`、`--output_nodes`进行脚本迁移。

> AST方案不支持ONNX模型文件迁移，ONNX文件迁移仅支持基于图结构的方案。

若省略`--output`与`--report`参数，MindConverter将在当前工作目录（Working directory）下自动创建`output`目录，将生成的脚本、转换报告、权重文件、权重映射表输出至该目录。

## 使用场景

MindConverter提供两种技术方案，以应对不同脚本迁移场景：

1. 用户希望迁移后脚本保持原脚本结构（包括变量、函数、类命名等与原脚本保持一致）；
2. 用户希望迁移后脚本保持较高的转换率，尽量少的修改、甚至不需要修改，即可实现迁移后模型脚本的执行。

对于上述第一种场景，推荐用户使用基于AST的方案进行转换（AST方案仅支持PyTorch脚本转换），AST方案通过对原PyTorch脚本的抽象语法树进行解析、编辑，将其替换为MindSpore的抽象语法树，再利用抽象语法树生成代码。理论上，AST方案支持任意模型脚本迁移，但语法树解析操作受原脚本用户编码风格影响，可能导致同一模型的不同脚本最终的转换率存在一定差异。

对于上述第二种场景，推荐用户使用基于图结构的脚本生成方案，计算图作为一种标准的模型描述语言，可以消除用户代码风格多样导致的脚本转换率不稳定的问题。在已支持算子的情况下，该方案可提供优于AST方案的转换率。

目前已基于计算机视觉领域典型模型对图结构的脚本转换方案进行测试。

> 1. 基于图结构的脚本生成方案，由于要以推理模式加载ONNX、TensorFlow模型，会导致转换后网络中Dropout算子丢失，需要用户手动补齐。
> 2. 基于图结构的脚本生成方案持续优化中。

## 使用示例

### 基于AST的脚本转换示例

若用户希望使用基于AST的方案进行脚本迁移，假设原PyTorch脚本路径为`/home/user/model.py`，希望将脚本输出至`/home/user/output`，转换报告输出至`/home/user/output/report`，则脚本转换命令为：

```text
mindconverter --in_file /home/user/model.py \
              --output /home/user/output \
              --report /home/user/output/report
```

转换报告中，对于未转换的代码行形式为如下，其中x, y指明的是原PyTorch脚本中代码的行、列号。对于未成功转换的算子，可参考[MindSporeAPI映射查询功能](https://www.mindspore.cn/docs/note/zh-CN/r1.3/index.html#operator_api) 手动对代码进行迁移。对于工具无法迁移的算子，会保留原脚本中的代码。

```text
line x:y: [UnConvert] 'operator' didn't convert. ...
```

转换报告示例如下所示：

```text
 [Start Convert]
 [Insert] 'import mindspore.ops as ops' is inserted to the converted file.
 line 1:0: [Convert] 'import torch' is converted to 'import mindspore'.
 ...
 line 157:23: [UnConvert] 'nn.AdaptiveAvgPool2d' didn't convert. Maybe could convert to mindspore.ops.operations.ReduceMean.
 ...
 [Convert Over]
```

对于部分未成功转换的算子，报告中会提供修改建议，如`line 157:23`，MindConverter建议将`torch.nn.AdaptiveAvgPool2d`替换为`mindspore.ops.operations.ReduceMean`。

### 基于图结构的脚本生成示例

#### TensorFlow模型脚本生成示例

使用TensorFlow模型脚本迁移，需要先将TensorFlow模型导出为pb格式，并且获取模型输入节点、输出节点名称。TensorFlow pb模型导出可参考[TensorFlow Pb模型导出教程](https://gitee.com/mindspore/mindinsight/blob/r1.3/mindinsight/mindconverter/docs/tensorflow_model_exporting_cn.md#)。

假设输入节点名称为`input_1:0`，输出节点名称为`predictions/Softmax:0`，模型输入样本尺寸为`1,224,224,3`，模型绝对路径为`xxx/frozen_model.pb`，希望将脚本、权重文件输出至`/home/user/output`，转换报告以及权重映射表输出至`/home/user/output/report`，则脚本生成命令为：

```text
mindconverter --model_file /home/user/xxx/frozen_model.pb --shape 1,224,224,3 \
              --input_nodes input_1:0 \
              --output_nodes predictions/Softmax:0 \
              --output /home/user/output \
              --report /home/user/output/report
```

执行该命令，MindSpore代码文件、权重文件、权重映射表和转换报告生成至相应目录。

由于基于图结构方案属于生成式方法，转换过程中未参考原TensorFlow脚本，因此生成的转换报告中涉及的代码行、列号均指生成后脚本。

另外对于未成功转换的算子，在代码中会相应的标识该节点输入、输出Tensor的shape（以`input_shape`, `output_shape`标识），便于用户手动修改。以Reshape算子为例，将生成如下代码：

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

通过`input_shape`、`output_shape`参数，用户可以十分便捷地完成算子替换，替换结果如下：

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

> 其中`--output`与`--report`参数可省略，若省略，该命令将在当前工作目录（Working directory）下自动创建`output`目录，将生成的脚本、转换报告输出至该目录。

映射表中分别保存算子在MindSpore中的权重信息（`converted_weight`）和在原始框架中的权重信息（`source_weight`）。

权重映射表示例如下所示：

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

#### ONNX模型文件生成示例

使用ONNX模型文件迁移，需要先从.onnx文件中获取模型输入节点、输出节点名称。获取ONNX模输入、输出节点名称，可使用 [Netron](https://github.com/lutzroeder/netron) 工具查看。

假设输入节点名称为`input_1:0`、输出节点名称为`predictions/Softmax:0`，模型输入样本尺寸为`1,3,224,224`，则可使用如下命令进行脚本生成：

```text
mindconverter --model_file /home/user/xxx/model.onnx --shape 1,3,224,224 \
              --input_nodes input_1:0 \
              --output_nodes predictions/Softmax:0 \
              --output /home/user/output \
              --report /home/user/output/report
```

执行该命令，MindSpore代码文件、权重文件、权重映射表和转换报告生成至相应目录。

由于基于图结构方案属于生成式方法，转换过程中未参考ONNX文件，因此生成的转换报告中涉及的代码行、列号均指生成后脚本。

另外，对于未成功转换的算子，在代码中会相应的标识该节点输入、输出Tensor的shape（以`input_shape`、`output_shape`标识），便于用户手动修改，示例见**TensorFlow模型脚本生成示例**。

## MindConverter错误码速查表

MindConverter错误码定义，请参考[链接](https://gitee.com/mindspore/mindinsight/blob/r1.3/mindinsight/mindconverter/docs/error_code_definition_cn.md# )。

## MindConverter支持的模型列表

[支持的模型列表（如下模型已基于x86 Ubuntu发行版，PyTorch 1.5.0以及TensorFlow 1.15.0测试通过）](https://gitee.com/mindspore/mindinsight/blob/r1.3/mindinsight/mindconverter/docs/supported_model_list_cn.md# )。

## 注意事项

1. 脚本转换工具本质上为算子驱动，对于MindConverter未维护的ONNX算子与MindSpore算子映射，将会出现相应的算子无法转换的问题，对于该类算子，用户可手动修改，或基于MindConverter实现映射关系，向MindInsight仓库贡献。
2. 在使用基于计算图的迁移时，MindConverter会根据`--shape`参数将模型输入的批次大小（batch size）、句子长度（sequence length）、图片尺寸（image shape）等尺寸相关参数固定下来，用户需要保证基于MindSpore重训练、推理时输入shape与转换时一致；若需要调整输入尺寸，请重新指定`--shape`进行转换或修改转换后脚本中涉及张量尺寸变更操作相应的操作数。
3. 脚本文件和权重文件输出于同一个目录下，转换报告和权重映射表输出于同一目录下。
4. 模型文件的安全性与一致性请用户自行保证。
