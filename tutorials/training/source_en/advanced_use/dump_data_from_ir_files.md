# Debugging with IR Diagrams

Translator: [xiaoxiaozhang](https://gitee.com/xiaoxinniuniu)

`Linux` `Ascend` `GPU` `Model Development` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Debugging with IR Diagrams](#debugging-with-ir-diagrams)
    - [Overview](#Overview)
    - [Generating IR Files](#Generating-ir-files)
    - [IR File Contents Introduction](#ir-file-contents-introduction)
    - [Dumping Required Data from the IR File](#dumping-required-data-from-the-ir-file)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/training/source_en/advanced_use/dump_data_from_ir_files.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

When a model compiled using MindSpore runs in the graph mode `context.set_context(mode=context.GRAPH_MODE)` and `context.set_context(save_graphs=True)` is set in the configuration, some intermediate files will be generated during graph compliation. These intermediate files are called IR files. Currently, there are three IR files:

- .ir file: An IR file that describes the model structure in text format and can be directly viewed using any text editors. We will also introduce how to view it in the following sections.

- .dat file: An IR file that describes the model structure more strictly than the .ir file. It contains more contents and can be directly viewed using any text editors.

- .dot file: An IR file that describes the topology relationships between different nodes. You can use this file by [graphviz](http://graphviz.org/) as the input to generate images for users to view the model structure. For models with multiple operators, it is recommended using the visualization component [MindInsight](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/dashboard.html#computational-graph-visualization) to visualize computing graphs.

In this tutorial, we use LeNet from ModelZoo as a demonstration in the Ascend environment. The related scripts can be found in [ModelZoo/LeNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet).

## Generating IR Files

Add the following code to `train.py`, When running the script, MindSpore will automatically store the IR files generated during compilation under the specified path.

```python
if __name__ == "__main__":
    context.set_context(save_graphs=True, save_graphs_path="path/to/ir/files")
```

In this tutorial, we run the training script on stand-alone computing device. When running on multiple computing devices, MindSpore will generate separate processes for each computing device. So, in multiple computing devices scenario, you are advised to read the ID of the current computing device from the training script and set an independent `save_graphs_path` for each decive to save the IR files to a different path. For example:

```python
device_id = os.getenv("DEVICE_ID")
context.set_context(save_graphs=True, save_graphs_path="path/to/ir/files"+device_id)
```

After the training command is executed, the following files are generated in the specified directory: the IR files starting with digits and underscores are generated during the ME diagram compilation. The calculation diagram is saved in each phase of the `pipeline`. Let's see the important phases, For examples, the `parse` phase parses the `construct` function of the entrance. The `symbol_resolve` phase recursively parses other functions and objects directly or indirectly referenced by the entry function. The `abstract_specialize` phase, type derivation and `shape` derivation are performed. The `optimize` phase, hardware-independent optimization is performed, The automatic differential and automatic parallel functions are also performed. The `validate` phase, the compiled compute graph is verified. The `task_emit` phase, the computing graph is transferred to the backend for further processing. The calculation graph is executed in the `execute` phase.

```text
.
├── 00_parse_[xxxx].ir
├── 00_parse.dat
├── 00_parse.dot
├── 01_symbol_resolve_[xxxx].ir
├── 01_symbol_resolve.dat
├── 01_symbol_resolve.dot
├── 02_combine_like_graphs_[xxxx].ir
├── 02_combine_like_graphs.dat
├── 02_combine_like_graphs.dot
├── 03_inference_opt_prepare_[xxxx].ir
├── 03_inference_opt_prepare.dat
├── 03_inference_opt_prepare.dot
├── 04_abstract_specialize_[xxxx].ir
├── 04_abstract_specialize.dat
├── 04_abstract_specialize.dot
├── 05_inline_[xxxx].ir
├── 05_inline.dat
├── 05_inline.dot
├── 06_py_pre_ad_[xxxx].ir
├── 06_py_pre_ad.dat
├── 06_py_pre_ad.dot
├── 07_pipeline_split_[xxxx].ir
├── 07_pipeline_split.dat
├── 07_pipeline_split.dot
├── 08_optimize_[xxxx].ir
├── 08_optimize.dat
├── 08_optimize.dot
├── 09_py_opt_[xxxx].ir
├── 09_py_opt.dat
├── 09_py_opt.dot
├── 10_validate_[xxxx].ir
├── 10_validate.dat
├── 10_validate.dot
├── 11_task_emit_[xxxx].ir
├── 11_task_emit.dat
├── 11_task_emit.dot
├── 12_execute_[xxxx].ir
├── 12_execute.dat
├── 12_execute.dot
...
```

## IR File Contents Introduction

The following is an example to describe the contents of the IR file.

```python
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context(save_graphs=True, save_graphs_path="./ir_files")

class Net(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x, y):
        x = x + y
        x = x * y
        return x

x = Tensor(3, mstype.float32)
y = Tensor(2, mstype.float32)
net = Net()
out = net(x, y)
print(out)
```

Use a text editing software (for example, vi) to open the `12_execute_[xxxx].ir` file. The file contents are as follows:

```text
  1 #IR entry      : @6_5_1_construct_wrapper.15
  2 #attrs         :
  3 check_set_strategy_valid_once_only : 1
  4 #Total params  : 2
  5
  6 %para1_x : <Tensor[Float32]x[const vector][]>
  7 %para2_y : <Tensor[Float32]x[const vector][]>
  8
  9 #Total subgraph : 1
 10
 11 subgraph attr:
 12 check_set_strategy_valid_once_only : 1
 13 subgraph @6_5_1_construct_wrapper.15() {
 14   %0([CNode]8) = Add(%para1_x, %para2_y) primitive_attrs: {output_names: [output], input_names: [x, y]}
 15       : (<Tensor[Float32]x[const vector][]>, <Tensor[Float32]x[const vector][]>) -> (<Tensor[Float32]x[const vector][]>)
 16       # In file /home/workspace/mindspore/mindspore/ops/composite/multitype_ops/add_impl.py(129)/    return F.tensor_add(x, y)/
 17       # In file demo.py(14)/        x = x + y/
 18   %1([CNode]10) = Mul(%0, %para2_y) primitive_attrs: {output_names: [output], input_names: [x, y]}
 19       : (<Tensor[Float32]x[const vector][]>, <Tensor[Float32]x[const vector][]>) -> (<Tensor[Float32]x[const vector][]>)
 20       # In file /home/workspace/mindspore/mindspore/ops/composite/multitype_ops/mul_impl.py(48)/    return F.tensor_mul(x, y)/
 21       # In file demo.py(15)/        x = x * y/
 22   return(%1)
 23       : (<Tensor[Float32]x[const vector][]>)
 24 }
 ```

The above contents can be divided into two parts, the first part is the input list and the second part is the graph structure. The first line tells us the name of the top MindSpore graph about the network, `@6_5_1_construct_wrapper.15`, or the entry graph. Line 4 tells us how many inputs are in the network. Line 6 to 7 are the input list, which is in the format of `%para[No.]_[name] : <[data_type]x[shape]>`. Line 9 tells us the number of subgraphs parsed by the network. Line 11 to 24 indicate the graph structure, which contains several nodes, namely, `CNode`. In this example, there are only two nodes: `Add` in row 14 and `Mul` in row 18.

The `CNode` information format is as follows: including the node name, attribute, input node, output information, format, and source code parsing call stack. The ANF diagram is a unidirectional acyclic graph. So, the connection between nodes is displayed only based on the input relationshape. The source code parsing call stack reflects the relationship between the `CNode` and the script source code. For example, line 20 is parsed from line 21, and line 21 corresponds to `x = x * y` of the script.

```text
  %[No.]([debug_name]) = [OpName]([arg], ...) primitive_attrs: {[key]: [value], ...}
      : (<[input data_type]x[input shape]>, ...) -> (<[output data_type]x[output shape]>, ...)
      # Call stack for source code parsing
```

><p style="font-family: Arial; font-size:0.9em;background: #EAEAAE;color:black;"><b>Notice:</b></p>
><p style="font-family: Arial; font-size:0.7em;background: #EAEAAE;color:black;">After several optimizations by the compiler, the node may undergo several changes (such as operator splitting and operator merging). The source code parsing call stack information of the node may not be in a one-to-one correspondence with the script. This is only an auxiliary method.</p>

## Dumping Required Data from the IR File

The following code comes from `lenet.py` in ModelZoo LeNet sample, Assume that you want to dump the first convolutional layer, that is, the `x = self.conv1(x)` data in the following code.

```python
class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
            self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))


    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

Generally, graph 0, `hwopt_d_end_graph_0_[xxxx].ir`, indicates the data subgraph (if the dataset_sink_mode is enabled), and graph 1 indicates the backbone network. So, search for `x = self.conv1(x)` in the dumped `hwopt_d_end_graph_1_[xxxx].ir`file, 4 results are obtained, 3 of them are `Cast` and `TransData`, skipping these operators generated by the precision conversion and format conversion optimization, we finally locate in lines 213 to 221, that is, `%24(equivoutput) = Conv2D(%23, %19)...`, corresponding to `conv1` in the network. In this way, you can obtain the op name (in the brackets of line 216, `Default/network-TrainOneStepWithLossScaleCell/network-WithLossCell/_backbone-LeNet5/conv1-Conv2d/Conv2D-op89`) in the compiled diagram from the following information.

```text
...
 213   %24(equivoutput) = Conv2D(%23, %19) {instance name: conv2d} primitive_attrs: {pri_format: NC1HWC0, stride: (1, 1, 1, 1), pad: (0, 0, 0, 0), pad_mode: valid, out_channel: 6, mode: 1     , dilation: (1, 1, 1, 1), output_names: [output], group: 1, format: NCHW, visited: true, offset_a: 0, kernel_size: (5, 5), groups: 1, input_names: [x, w], pad_list: (0, 0, 0, 0), IsF     eatureMapOutput: true, IsFeatureMapInputList: (0)}
 214       : (<Tensor[Float16]x[const vector][32, 1, 32, 32]>, <Tensor[Float16]x[const vector][6, 1, 5, 5]>) -> (<Tensor[Float16]x[const vector][32, 6, 28, 28]>)
 215       : (<Float16xNC1HWC0[const vector][32, 1, 32, 32, 16]>, <Float16xFracZ[const vector][25, 1, 16, 16]>) -> (<Float16xNC1HWC0[const vector][32, 1, 28, 28, 16]>)
 216       : (Default/network-TrainOneStepWithLossScaleCell/network-WithLossCell/_backbone-LeNet5/conv1-Conv2d/Conv2D-op89)
 217       # In file /home/workspace/mindspore/build/package/mindspore/nn/layer/conv.py(263)/        output = self.conv2d(x, self.weight)/
 218       # In file /home/workspace/mindspore/model_zoo/official/cv/lenet/src/lenet.py(49)/        x = self.conv1(x)/
 219       # In file /home/workspace/mindspore/build/package/mindspore/train/amp.py(101)/            out = self._backbone(data)/
 220       # In file /home/workspace/mindspore/build/package/mindspore/nn/wrap/loss_scale.py(323)/        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)/
 221       # In file /home/workspace/mindspore/build/package/mindspore/train/dataset_helper.py(87)/            return self.network(*outputs)/
...
```

After obtaining the op name, we can execute the dump process to save the input and output of the operator for debugging. Here, we will introduce a method called synchronous dump.

1.Create the configuration file `data_dump.json`, this file stores the operators information to be dumped, copy the op name obtained from previous step to the `kernels` key. For details about this file, see the [custom debugging info](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/custom_debugging_info.html#asynchronous-dump).

```text
{
    "common_dump_settings": {
        "dump_mode": 1,
        "path": "/absolute_path",
        "net_name": "LeNet",
        "iteration": 0,
        "input_output": 0,
        "kernels": ["Default/network-TrainOneStepWithLossScaleCell/network-WithLossCell/_backbone-LeNet5/conv1-Conv2d/Conv2D-op89"],
        "support_device": [0,1,2,3,4,5,6,7]
    },
    "e2e_dump_settings": {
        "enable": true,
        "trans_flag": false
    }
}
```

2.Configure environment variables and specify the path of the configuration file.

```bash
export MINDSPORE_DUMP_CONFIG={Absolute path of data_dump.json}
```

3.Execute the case to dump data. During the execution, MindSpore dumps the input and output data of a specified operator to the specified path.

In this example, the following files are obtained, which correspond to the input and output of the operator.

```text.
├── Default--network-TrainOneStepWithLossScaleCell--network-WithLossCell--_backbone-LeNet5--conv1-Conv2d--Conv2D-op89_input_0_shape_32_1_32_32_16_Float16_NC1HWC0.bin
├── Default--network-TrainOneStepWithLossScaleCell--network-WithLossCell--_backbone-LeNet5--conv1-Conv2d--Conv2D-op89_input_1_shape_25_1_16_16_Float16_FracZ.bin
└── Default--network-TrainOneStepWithLossScaleCell--network-WithLossCell--_backbone-LeNet5--conv1-Conv2d--Conv2D-op89_output_0_shape_32_1_28_28_16_Float16_NC1HWC0.bin
```

4.Parse the dump data.

You can use numpy.fromfile to read the file generated in previous step. The ndarray obtained after reading is the input/output of the corresponding operator.

```python
import numpy
output = numpy.fromfile("Default--network-TrainOneStepWithLossScaleCell--network-WithLossCell--_backbone-LeNet5--conv1-Conv2d--Conv2D-op89_input_0_shape_32_1_32_32_16_Float16_NC1HWC0.bin")
print(output)
```

The output is:

```text
[1.17707155e-17 4.07526143e-17 5.84038559e-18 ... 0.00000000e+00 0.00000000e+00 0.00000000e+00]
```
