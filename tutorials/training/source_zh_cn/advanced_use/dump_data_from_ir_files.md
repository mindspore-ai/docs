# 借助IR图进行调试

`Linux` `Ascend` `GPU` `模型开发` `初级` `中级` `高级`

<!-- TOC -->

- [借助IR图进行调试](#借助ir图进行调试)
    - [概述](#概述)
    - [生成IR文件](#生成ir文件)
    - [IR文件内容介绍](#ir文件内容介绍)
    - [从IR文件中Dump出想要的数据](#从ir文件中dump出想要的数据)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/training/source_zh_cn/advanced_use/dump_data_from_ir_files.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

在图模式`context.set_context(mode=context.GRAPH_MODE)`下运行用MindSpore编写的模型时，若配置中设置了`context.set_context(save_graphs=True)`，运行时会输出一些图编译过程中生成的一些中间文件，我们称为IR文件，当前主要有三种格式的IR文件：

- ir后缀结尾的IR文件：一种比较直观易懂的以文本格式描述模型结构的文件，可以直接用文本编辑软件查看。在下文中我们也将介绍此文件的查看方式。
- dat后缀结尾的IR文件：一种相对于ir后缀结尾的文件格式定义更为严谨的描述模型结构的文件，包含的内容更为丰富，可以直接用文本编辑软件查看。
- dot后缀结尾的IR文件：描述了不同节点间的拓扑关系，可以用[graphviz](http://graphviz.org)将此文件作为输入生成图片，方便用户直观地查看模型结构，不过对于算子比较多的模型，我们不推荐通过此方法查看模型结构。

在本教程中，我们使用ModelZoo中的LeNet在Ascend环境上作为示范。相关的脚本可以在[ModelZoo/LeNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet)找到。

## 生成IR文件

在`train.py`中，我们在`set_context`函数中添加如下代码，运行训练脚本时，MindSpore会自动将编译过程中产生的IR文件存放到指定路径。

```python
if __name__ == "__main__":
    context.set_context(save_graphs=True, save_graphs_path="path/to/ir/files")
```

 在本教程中，我们运行的为单机版本的训练脚本。当运行的脚本使用多个计算设备时，MindSpore会为每一个计算设备生成一个独立的进程。因此我们建议用户在多卡版本的训练脚本中读取当前的计算设备id，从而为每个设备设置独立的`save_graphs_path`实现将每个设备的IR文件保存在不同的路径下。例如：

```python
device_id = os.getenv("DEVICE_ID")
context.set_context(save_graphs=True, save_graphs_path="path/to/ir/files"+device_id)
```

执行训练命令后，在指定的目录生成如下文件。其中以数字下划线开头的IR文件是在ME编译图过程中输出的，`pipeline`各阶段分别会保存一次计算图。下面介绍比较重要的阶段，例如`parse`阶段会解析入口的`construct`函数；`symbol_resolve`阶段会递归解析入口函数直接或间接引用到的其他函数和对象；`abstract_specialize`阶段会做类型推导和`shape`推导；`optimize`阶段主要是进行和硬件无关的优化，自动微分与自动并行功能也是在该阶段展开；`validate`阶段会校验编译出来的计算图；`task_emit`阶段将计算图传给后端进一步处理；`execute`阶段会执行该计算图。

```bash
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

## IR文件内容介绍

下面以一个简单的例子来说明IR文件的内容。

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

使用文本编辑软件（例如vi）打开文件`12_execute_[xxxx].ir`，内容如下所示：

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

以上内容可分为两个部分，第一部分为输入列表，第二部分为图结构。 其中第1行告诉了我们该网络的顶图名称`@6_5_1_construct_wrapper.15`，也就是入口图。 第4行告诉了我们该网络有多少个输入。 第6-7行为输入列表，遵循`%para[序号]_[name] : <[data_type]x[shape]>`的格式。 第9行告诉我们该网络解析出来的子图数量。 第11-24行为图结构，含有若干节点，即`CNode`。该示例中只有2个节点,分别为14行的`Add`和18行的`Mul`。

`CNode`的信息遵循如下格式，包含节点名称、属性、输入节点、输出信息、格式、源码解析调用栈等信息，由于ANF图为单向无环图，所以这里仅根据输入关系体现节点与节点的连接关系。源码解析调用栈则体现了`CNode`与脚本源码之间的关系，例如第20行由第21行解析而来，而第21行能对应到脚本的`x = x * y`。

```text
  %[序号]([debug_name]) = [OpName]([arg], ...) primitive_attrs: {[key]: [value], ...}
      : (<[输入data_type]x[输入shape]>, ...) -> (<[输出data_type]x[输出shape]>, ...)
      # 源码解析调用栈
```

> 需要注意的是经过编译器的若干优化处理后，节点可能经过了若干变幻（如算子拆分、算子融合等），节点的源码解析调用栈信息与脚本可能无法完全一一对应，这里仅作为辅助手段。

## 从IR文件中Dump出想要的数据

下面的代码片段来自ModelZoo中LeNet示例中的`lenet.py`， 假设我们想要Dump出第一个卷积层也就是下述代码片段中的`x = self.conv1(x)`的数据。

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

一般而言，后端的0图代表数据子图（若开启了数据下沉模式），1图代表主干网络，所以这里我们在Dump出来的`hwopt_d_end_graph_1_[xxxx].ir`文件中搜索`x = self.conv1(x)`，会得到4处结果，其中有3处为`Cast`和`TransData`。 越过该类精度转换、格式转换优化产生的`Cast`和`TransData`，我们最终定位到第213-221行，`%24(equivoutput) = Conv2D(%23, %19)...`，此处即对应网络中的`conv1`。从而在下方的信息中得到该算子在所编译的图中对应的op名（第216行的括号内，`Default/network-TrainOneStepWithLossScaleCell/network-WithLossCell/_backbone-LeNet5/conv1-Conv2d/Conv2D-op89`）。

```bash
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

得到算子的op名称之后，我们就可以执行Dump流程来保存算子的输入输出方便调试了。在这里我们介绍一种叫做同步Dump的方法。

1. 创建配置文件`data_dump.json`，该文件保存了需要Dump的算子信息，将我们在上一步中定位到的op名称复制到`kernels`键对应的列表内，关于该文件更多的信息，可以参考[自定义调试信息](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/custom_debugging_info.html#id5)。

   ```json
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

2. 配置环境变量，指定配置文件的路径。

   ```bash
   export MINDSPORE_DUMP_CONFIG={Absolute path of data_dump.json}
   ```

3. 执行用例Dump数据，在运行中MindSpore会Dump指定算子的输入输出数据到指定路径。

   在本例中，最后我们得到了如下文件，分别对应着该算子的输入和输出。

   ```bash
   .
   ├── Default--network-TrainOneStepWithLossScaleCell--network-WithLossCell--_backbone-LeNet5--conv1-Conv2d--Conv2D-op89_input_0_shape_32_1_32_32_16_Float16_NC1HWC0.bin
   ├── Default--network-TrainOneStepWithLossScaleCell--network-WithLossCell--_backbone-LeNet5--conv1-Conv2d--Conv2D-op89_input_1_shape_25_1_16_16_Float16_FracZ.bin
   └── Default--network-TrainOneStepWithLossScaleCell--network-WithLossCell--_backbone-LeNet5--conv1-Conv2d--Conv2D-op89_output_0_shape_32_1_28_28_16_Float16_NC1HWC0.bin
   ```

4. 解析Dump的数据。

   可以通过`numpy.fromfile`读取上一步生成的文件。读取后得到的`ndarray`即对应该算子的输入/输出。

   ```python
   import numpy
   output = numpy.fromfile("Default--network-TrainOneStepWithLossScaleCell--network-WithLossCell--_backbone-LeNet5--conv1-Conv2d--Conv2D-op89_input_0_shape_32_1_32_32_16_Float16_NC1HWC0.bin")
   print(output)
   ```

   输出为：

   ```text
   [1.17707155e-17 4.07526143e-17 5.84038559e-18 ... 0.00000000e+00 0.00000000e+00 0.00000000e+00]
   ```
