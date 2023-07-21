# 使用工具迁移第三方框架脚本

`Linux` `Ascend` `模型开发` `初级`

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_zh_cn/advanced_use/migrate_3rd_scripts_mindconverter.md)

## 概述

MindConverter是一款将PyTorch模型脚本转换至MindSpore的脚本迁移工具。结合转换报告的提示信息，用户对转换后脚本进行微小改动，即可快速将PyTorch模型脚本迁移至MindSpore。

## 安装

此工具为MindInsight的子模块，安装MindInsight后，即可使用MindConverter，MindInsight安装请参考该[安装文档](https://www.mindspore.cn/install/)。

## 用法

MindConverter提供命令行（Command-line interface, CLI）的使用方式，命令如下。

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

**MindConverter提供两种模型脚本迁移方案：**

1. **基于抽象语法树(Abstract syntax tree, AST)的脚本转换**：指定`--in_file`的值，将使用基于AST的脚本转换方案；
2. **基于图结构的脚本生成**：指定`--model_file`与`--shape`将使用基于图结构的脚本生成方案。

> 若同时指定了`--in_file`，`--model_file`将默认使用AST方案进行脚本迁移。

当使用基于图结构的脚本生成方案时，要求必须指定`--shape`的值；当使用基于AST的脚本转换方案时，`--shape`会被忽略。

其中，`--output`与`--report`参数可省略。若省略，MindConverter将在当前工作目录（Working directory）下自动创建`output`目录，将生成的脚本、转换报告输出至该目录。

另外，当使用基于图结构的脚本生成方案时，请确保原PyTorch项目已在Python包搜索路径中，可通过CLI进入Python交互式命令行，通过import的方式判断是否已满足；若未加入，可通过`--project_path`命令手动将项目路径传入，以确保MindConverter可引用到原PyTorch脚本。

> 假设用户项目目录为`/home/user/project/model_training`，用户可通过如下命令手动项目添加至包搜索路径中：`export PYTHONPATH=/home/user/project/model_training:$PYTHONPATH`
> 此处MindConverter需要引用原PyTorch脚本，是因为PyTorch模型反向序列化过程中会引用原脚本。

## 使用场景

MindConverter提供两种技术方案，以应对不同脚本迁移场景：

1. 用户希望迁移后脚本保持原有PyTorch脚本结构（包括变量、函数、类命名等与原脚本保持一致）；
2. 用户希望迁移后脚本保持较高的转换率，尽量少的修改、甚至不需要修改，即可实现迁移后模型脚本的执行。

对于上述第一种场景，推荐用户使用基于AST的方案进行转换，AST方案通过对原PyTorch脚本的抽象语法树进行解析、编辑，将其替换为MindSpore的抽象语法树，再利用抽象语法树生成代码。理论上，AST方案支持任意模型脚本迁移，但语法树解析操作受原脚本用户编码风格影响，可能导致同一模型的不同脚本最终的转换率存在一定差异。

对于上述第二种场景，推荐用户使用基于图结构的脚本生成方案，计算图作为一种标准的模型描述语言，可以消除用户代码风格多样导致的脚本转换率不稳定的问题。在已支持算子的情况下，该方案可提供优于AST方案的转换率。

目前已基于典型图像分类网络(Resnet, VGG)对图结构的脚本转换方案进行测试。

> 1. 基于图结构的脚本生成方案，目前仅支持单输入、单输出模型，对于多输入模型暂不支持；
> 2. 基于图结构的脚本生成方案，由于要基于推理模式加载PyTorch模型，会导致转换后网络中Dropout算子丢失，需要用户手动补齐；
> 3. 基于图结构的脚本生成方案持续优化中。

## 使用示例

### 基于AST的脚本转换示例

若用户希望使用基于AST的方案进行脚本迁移，假设原PyTorch脚本路径为`/home/user/model.py`，希望将脚本输出至`/home/user/output`，转换报告输出至`/home/user/output/report`，则脚本转换命令为：

```bash
mindconverter --in_file /home/user/model.py \
              --output /home/user/output \
              --report /home/user/output/report
```

转换报告中，对于未转换的代码行形式为如下，其中x, y指明的是原PyTorch脚本中代码的行、列号。对于未成功转换的算子，可参考[MindSporeAPI映射查询功能](https://www.mindspore.cn/doc/note/zh-CN/r1.0/index.html#operator_api) 手动对代码进行迁移。对于工具无法迁移的算子，会保留原脚本中的代码。

```text
line x:y: [UnConvert] 'operator' didn't convert. ...
```

转换报告示例如下所示：

```text
 [Start Convert]
 [Insert] 'import mindspore.ops.operations as P' is inserted to the converted file.
 line 1:0: [Convert] 'import torch' is converted to 'import mindspore'.
 ...
 line 157:23: [UnConvert] 'nn.AdaptiveAvgPool2d' didn't convert. Maybe could convert to mindspore.ops.operations.ReduceMean.
 ...
 [Convert Over]
```

对于部分未成功转换的算子，报告中会提供修改建议，如`line 157:23`，MindConverter建议将`torch.nn.AdaptiveAvgPool2d`替换为`mindspore.ops.operations.ReduceMean`。

### 基于图结构的脚本生成示例

若用户已将PyTorch模型保存为.pth格式，假设模型绝对路径为`/home/user/model.pth`，该模型期望的输入样本shape为(3, 224, 224)，原PyTorch脚本位于`/home/user/project/model_training`，希望将脚本输出至`/home/user/output`，转换报告输出至`/home/user/output/report`，则脚本生成命令为：

```bash
mindconverter --model_file /home/user/model.pth --shape 3,224,224 \
              --output /home/user/output \
              --report /home/user/output/report \
              --project_path /home/user/project/model_training
```

执行该命令，MindSpore代码文件、转换报告生成至相应目录。

基于图结构的脚本生成方案产生的转换报告格式与AST方案相同。然而，由于基于图结构方案属于生成式方法，转换过程中未参考原PyTorch脚本，因此生成的转换报告中涉及的代码行、列号均指生成后脚本。

另外对于未成功转换的算子，在代码中会相应的标识该节点输入、输出Tensor的shape（以`input_shape`, `output_shape`标识），便于用户手动修改。以Reshape算子为例（暂不支持Reshape），将生成如下代码：

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

## 注意事项

1. PyTorch不作为MindInsight明确声明的依赖库。若想使用基于图结构的脚本生成工具，需要用户手动安装与生成PyTorch模型版本一致的PyTorch库（MindConverter推荐使用PyTorch 1.4.0或PyTorch 1.6.0进行脚本生成）；
2. 脚本转换工具本质上为算子驱动，对于MindConverter未维护的PyTorch或ONNX算子与MindSpore算子映射，将会出现相应的算子无法转换的问题，对于该类算子，用户可手动修改，或基于MindConverter实现映射关系，向MindInsight仓库贡献。
