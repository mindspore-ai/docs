# 借助IR图进行调试

`Linux` `Ascend` `GPU` `模型开发` `初级` `中级` `高级`

<!-- TOC -->

- [借助IR图进行调试](#借助ir图进行调试)
    - [概述](#概述)
    - [生成IR文件](#生成ir文件)
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

执行训练命令后，在指定的目录生成如下文件，其中以数字下划线开头的IR文件是MindSpore在ME pipeline不同阶段输出的IR文件，以`hwopt`开头的则是后端LLO不同优化阶段输出的IR文件。其余的则如文件名所示，并根据子图的个数分别保存。

```bash
.
├── 00_parse_0160.ir
├── 00_parse.dat
├── 00_parse.dot
├── 01_symbol_resolve_01161.ir
├── 01_symbol_resolve.dat
├── 01_symbol_resolve.dot
├── 02_combine_like_graphs_0162.ir
├── 02_combine_like_graphs.dat
├── 02_combine_like_graphs.dot
├── 03_inference_opt_prepare_0163.ir
├── 03_inference_opt_prepare.dat
├── 03_inference_opt_prepare.dot
├── 04_abstract_specialize_0164.ir
...
├── hwopt_d_end_graph_0_0154.ir
├── hwopt_d_end_graph_1_0533.ir
├── hwopt_d_end_graph_2_0665.ir
├── hwopt_d_end_graph_3_0599.ir
...
```

## 从IR文件中Dump出想要的数据

下面的代码片段来自ModelZoo中LeNet示例中的`lenet.py`， 假设我们想要Dump出第一个卷积层也就是下述代码片段中的`conv1`的数据。

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

一般而言，后端的0图代表数据子图（若开启了数据下沉），1图代表主干网络，所以这里我们使用Dump出来的`hwopt_d_end_graph_1_0533.ir`作为示例。在文件开头的参数列表中我们可以发现`%para4_conv1.weight`对应着我们想要Dump的算子的输入。搜索`%para4_conv1.weight`，我们可以定位到124行：`%18(equivoutput) = Cast(%para4_conv1.weight)`，再搜索`%18`，定位到129行：`%19(equivoutput) = TransData(%18)`。越过这类精度转换、格式转换优化产生的`Cast`和`TransData`算子，搜索`%19`，我们最终定位到153行：`%24(equivoutput) = Conv2D(%23, %19)`，该算子即为网络中的`conv1`。从而在下方的信息中得到该算子在所编译的图中对应的op名（第156行）。

```bash
...
7: %para3_current_iterator_step : <Ref[Tensor(I32)]x[const vector][]>  :  <Int32xDefaultFormat[const vector][]>  :  IsWeight:true
8: %para4_conv1.weight : <Ref[Tensor(F32)]x[const vector][6, 1, 5, 5]>  :  <Float32xDefaultFormat[const vector][6, 1, 5, 5]>  :  IsWeight:true
9: %para5_conv2.weight : <Ref[Tensor(F32)]x[const vector][16, 6, 5, 5]>  :  <Float32xDefaultFormat[const vector][16, 6, 5, 5]>  :  IsWeight:true
10: %para6_fc1.weight : <Ref[Tensor(F32)]x[const vector][120, 400]>  :  <Float32xDefaultFormat[const vector][120, 400]>  :  IsWeight:true
11: %para7_fc1.bias : <Ref[Tensor(F32)]x[const vector][120]>  :  <Float32xDefaultFormat[const vector][120]>  :  IsWeight:true
...
124: %18(equivoutput) = Cast(%para4_conv1.weight) primitive_attrs: {pri_format: NC1HWC0, IsFeatureMapOutput: false, output_names: [output], input_names: (x), DstT: Float16, dst_type: Fl oat16, IsFeatureMapInputList: (), SrcT: Float32, is_backed_cast: false}
125: : (<Ref[Tensor(F32)]x[const vector][6, 1, 5, 5]>) -> (<Tensor[Float16]x[const vector][6, 1, 5, 5]>)
126: : (<Float32xDefaultFormat[const vector][6, 1, 5, 5]>) -> (<Float16xDefaultFormat[const vector][6, 1, 5, 5]>)
127: : (Default/network-TrainOneStepWithLossScaleCell/network-WithLossCell/_backbone-LeNet5/conv1-Conv2d/Cast-op90)
128: # In file /home/workspace/mindspore/build/package/mindspore/nn/layer/conv.py(253)/ output = self.conv2d(x, self.weight)/
129 %19(equivoutput) = TransData(%18) primitive_attrs: {dst_format: FracZ, IsFeatureMapInputList: (), pri_format: NC1HWC0, datadump_original_names: (), IsFeatureMapOutput: false, src format: NCHW}
130: : (<Tensor[Float16]x[const vector][6, 1, 5, 5]>) -> (<Tensor[Float16]x[const vector][6, 1, 5, 5]>)
131: : (<Float16xDefaultFormat[const vector][6, 1, 5, 5]>) -> (<Float16xFracZ[const vector][25, 1, 16, 16]>)
132: : (Default/TransData-op447)
133: # In file /home/workspace/mindspore/build/package/mindspore/nn/layer/conv.py(253)/ output = self.conv2d(x, self.weight)/
...
153: %24(equivoutput) = Conv2D(%23, %19) {instance name: conv2d} primitive_attrs: {pri_format: NC1HWC0, stride: (1, 1, 1, 1), pad: (0, 0, 0, 0), pad_mode: valid, out_channel: 6, mode: 1 , dilation: (1, 1, 1, 1), output_names: [output], group: 1, format: NCHW, offset_a: 0, kernel_size: (5, 5), groups: 1, input_names: [x, w], pad_list: (0, 0, 0, 0), IsFeatureMapOutput : true, IsFeatureMapInputList: (0)}
154: : (<Tensor[Float16]x[const vector][32, 1, 32, 32]>, <Tensor[Float16]x[const vector][6, 1, 5, 5]>) -> (<Tensor[Float16]x[const vector][32, 6, 28, 28]>)
155: : (<Float16xNC1HWC0[const vector][32, 1, 32, 32, 16]>, <Float16xFracZ[const vector][25, 1, 16, 16]>) -> (<Float16xNC1HWC0[const vector][32, 1, 28, 28, 16]>)
156: : (Default/network-TrainOneStepWithLossScaleCell/network-WithLossCell/_backbone-LeNet5/conv1-Conv2d/Conv2D-op89)
157: # In file /home/workspace/mindspore/build/package/mindspore/nn/layer/conv.py(253)/ output = self.conv2d(x, self.weight)/
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
