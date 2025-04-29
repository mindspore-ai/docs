# IR文件分析

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/debug/error_analysis/mindir.md)

## 概述

在图模式`set_context(mode=GRAPH_MODE)`下运行用MindSpore编写的模型时，若设置了环境变量`MS_DEV_SAVE_GRAPHS`的值为2，运行时会输出一些图编译过程中生成的中间文件，称为IR文件。当前主要有两种格式的IR文件：

- ir后缀结尾的IR文件：一种比较直观易懂的以文本格式描述模型结构的文件，可以直接用文本编辑软件查看。
- dot后缀结尾的IR文件：若设置了环境变量`MS_DEV_SAVE_GRAPHS`的值为3, 运行时会输出后缀为dot的ir文件。该文件描述了不同节点间的拓扑关系，可以用[graphviz](http://graphviz.org)将此文件作为输入生成图片，方便用户直观地查看模型结构。

## 如何保存IR

通过设置环境变量`MS_DEV_SAVE_GRAPHS`的值为2来保存各个编译阶段的中间代码。被保存的中间代码有两种格式，默认保存后缀名为`.ir`的文本格式的ir文件。如果设置环境变量`MS_DEV_SAVE_GRAPHS`的值为3会打印后缀名为`.dot`的图形化格式的ir文件。当网络规模不大时，建议使用更直观的图形化格式来查看，当网络规模较大时建议使用更高效的文本格式来查看。

`.dot`文件可以通过graphviz转换为图片格式来查看，例如将dot转换为png的命令是`dot -Tpng *.dot -o *.png`。

在训练脚本`train.py`中，添加如下代码，运行训练脚本时，MindSpore会自动将编译过程中产生的IR文件存放到指定路径。

```python
import os
os.environ['MS_DEV_SAVE_GRAPHS'] = "3"
os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = "path/to/ir/files"
```

执行训练命令后，在指定的路径下生成了若干个文件：

```text
.
├──00_bootstrap_0000.ir
├──00_bootstrap_0001.dot
├──01_type_inference_0002.ir
├──01_type_inference_0003.dot
├──02_graph_reusing_0004.ir
├──02_graph_reusing_0005.dot
├──03_auto_monad_0006.ir
├──03_auto_monad_0007.dot
...
```

其中以数字下划线开头的IR文件是在前端编译图过程中生成的，编译过程中各阶段分别会保存一次计算图。下面介绍图编译过程中比较重要的阶段:

- `bootstrap`阶段负责解析入口函数，该阶段会初步生成MindIR，如果查看IR文件，可以观察到该一个基础的解析节点，代表图的入口函数，以及一个相应的有必需参数的调用节点。
- `type_inference`阶段负责类型推导和符号解析。该阶段递归地解析程序的入口函数，解析对其他函数和对象的引用，并推断所有节点的数据类型和形状信息。与不支持的语法或未解决的引用相关的错误会在这个阶段被标记出来，为开发者提供早期反馈。
- `optimize`阶段负责硬件无关的优化，自动微分与自动并行功能也是在该阶段展开。该阶段又可细分为若干个子阶段，在IR文件列表中，其中以`opt_pass_[序号]`为前缀的文件分别是这些子阶段结束后保存的IR文件，非框架开发人员无需过多关注；
- `validate`阶段负责校验编译出来的计算图，如果到此阶段IR中还有仅临时使用的内部算子，则会报错退出；
- `task_emit`阶段负责将计算图传给后端进一步处理；
- `execute`阶段负责启动执行图流程，该阶段的IR图是前端编译阶段的最终图。

此外，后端由于比较贴近底层，后端优化过程中保存的其他IR文件（如以`hwopt`开头的文件）非框架开发人员也无需过多关注。非框架开发人员仅需查看名为`graph_build_[图序号]_[IR文件序号].ir`的文件，即经过前后端全部优化后的IR。

由于IR文件序号放在文件末尾，按照文件名排序时，IR文件往往不是按照ir生成顺序排序的。若想以文件生成顺序排列IR文件，可以使用Linux的awk命令`find ./ -name '*ir' | awk --field-separator="_" '{print $(NF) "--->" $0}' | sort -n`。

> 由于后端以子图为单位进行优化，故可能会保存多份文件，与前端多个子图都保存在同一文件中的机制不同。

## IR文件解读

下面以一个简单的例子来说明IR文件的内容，运行该脚本：

```python
import os
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops

ms.set_context(mode=ms.GRAPH_MODE)
os.environ['MS_DEV_SAVE_GRAPHS'] = '2'
os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = './ir'

class Net(nn.Cell):
    def __init__(self):
        super().__init__()

    def func(x, y):
        return ops.div(x, y)

    def construct(self, x, y):
        a = ops.sub(x, 1)
        b = ops.add(a, y)
        if b :
            b = ops.mul(b, self.func(a, b))
        return b

input1 = ms.Tensor(3, ms.float32)
input2 = ms.Tensor(2, ms.float32)
net = Net()
out = net(input1, input2)
print(out)
```

### ir文件介绍

使用文本编辑软件（例如`vi`）打开执行完后输出的IR文件`18_execute_0161.ir`，内容如下所示（此处版本为MindSpore 2.3，后续版本中内容可能会有一些细微变化）：

```text
  1 # IR entry: @19_1___main___Net_construct_304
  2 # Total subgraphs: 3
  3
  4 # Attrs:
  5 has_shard: 0
  6 has_attached: 1
  7 jit_level:
  8 check_set_strategy_valid_once_only: 1
  9 FLASH_SP_RUN_ONCE_ONLY: 1
 10 pynative_run_in_graph: 0
 11 less_bn: 0
 12 auto_parallel_finish_pre_action: 1
 13
 14 # Total params: 2
 15 # Params:
 16 %para1_x: <Tensor[Float32], ()> : []
 17 %para2_y: <Tensor[Float32], ()> : []
 18
 19 Node counting information:
 20 Total number of nodes: 29
 21 Total number of cnodes: 12
 22
 23 subgraph attr:
 24 has_shard: 0
 25 has_attached: 1
 26 jit_level:
 27 check_set_strategy_valid_once_only: 1
 28 FLASH_SP_RUN_ONCE_ONLY: 1
 29 pynative_run_in_graph: 0
 30 less_bn: 0
 31 auto_parallel_finish_pre_action: 1
 32 subgraph instance: 19_1___main___Net_construct_304 : 0x135400418
 33 # In file t6.py:15~20, 4~16/    def construct(self, x, y):/
 34 subgraph @19_1___main___Net_construct_304() {
 35   %0(CNode_310$a) = PrimFunc_Sub(%para1_x, Tensor(shape=[], dtype=Float32, value=1)) cnode_attrs: {checkpoint: Bool(1), is_dynamic_len: Bool(0)}
 36       : (<Tensor[Float32], ()>, <Tensor[Float32], (), value=...>) -> (<Tensor[Float32], ()>)
 37       # Fullname with scope: (Default/Sub-op1)
 38       # In file t6.py:15~20, 4~16/    def construct(self, x, y):/
 39       # In file t6.py:16, 12~25/        a = ops.sub(x, 1)/
 40       # In file t6.py:16, 12~19/        a = ops.sub(x, 1)/<~~This line of code can be shared by multiple nodes, and may be duplicated./
 41       # In file /workspace/mindspore/build/package/mindspore/ops/auto_generate/gen_ops_def.py:5251~5294, 0~31/def sub(input, other):/
 42       # In file /workspace/mindspore/build/package/mindspore/ops/auto_generate/gen_ops_def.py:5294, 11~31/    return sub_op(input, other)/
 43   %1(CNode_309$b) = PrimFunc_Add(%0, %para2_y) cnode_attrs: {checkpoint: Bool(1), is_dynamic_len: Bool(0)}
 44       : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
 45       # Fullname with scope: (Default/Add-op1)
 46       # In file t6.py:15~20, 4~16/    def construct(self, x, y):/
 47       # In file t6.py:17, 12~25/        b = ops.add(a, y)/
 48       # In file t6.py:17, 12~19/        b = ops.add(a, y)/<~~This line of code can be shared by multiple nodes, and may be duplicated./
 49       # In file /workspace/mindspore/build/package/mindspore/ops/auto_generate/gen_ops_def.py:183~241, 0~31/def add(input, other):/
 50       # In file /workspace/mindspore/build/package/mindspore/ops/auto_generate/gen_ops_def.py:241, 11~31/    return add_op(input, other)/
 51   %2(CNode_308) = PrimFunc_Cast(%1, I64(30)) primitive_attrs: {output_names: [output], input_names: [x, dst_type]} cnode_attrs: {checkpoint: Bool(1), is_dynamic_len: Bool(0)}
 52       : (<Tensor[Float32], ()>, <Int64, NoShape>) -> (<Tensor[Bool], ()>)
 53       # Fullname with scope: (Default/Cast-op1)
 54       # In file /workspace/mindspore/build/package/mindspore/_extends/parse/standard_method.py:2747~2749, 0~23/def bool_(x):/
 55       # In file /workspace/mindspore/build/package/mindspore/_extends/parse/standard_method.py:2749, 11~23/    return x.__bool__()/
 56       # In file /workspace/mindspore/build/package/mindspore/_extends/parse/standard_method.py:2749, 11~21/    return x.__bool__()/<~~This line of code can be shared by multiple nodes, and may be duplicated./
 57       # In file /workspace/mindspore/build/package/mindspore/_extends/parse/standard_method.py:3267~3272, 0~34/def tensor_bool(x):/
 58       # In file /workspace/mindspore/build/package/mindspore/_extends/parse/standard_method.py:3270~3271, 4~38/    if is_cond and F.isconstant(x):/
 59       # In file /workspace/mindspore/build/package/mindspore/_extends/parse/standard_method.py:3272, 11~34/    return F.cast(x, mstype.bool_)/<~~This line of code can be shared by multiple nodes, and may be duplicated./
 60   %3(CNode_317) = Partial(@20_4_✓__main___Net_construct_311, %1, %0) primitive_attrs: {side_effect_propagate: I64(1)} cnode_attrs: {checkpoint: Bool(1)}
 61       : (<Func, NoShape>, <Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Func, NoShape>)
 62       # Fullname with scope: (Default/Partial-op0)
 63   %4(CNode_316) = Partial(@21_14_✗__main___Net_construct_314, %1) primitive_attrs: {side_effect_propagate: I64(1)} cnode_attrs: {checkpoint: Bool(1)}
 64       : (<Func, NoShape>, <Tensor[Float32], ()>) -> (<Func, NoShape>)
 65       # Fullname with scope: (Default/Partial-op1)
 66   %5(ValueNode_307) = Switch(%2, %3, %4) cnode_attrs: {checkpoint: Bool(1)}
 67       : (<Tensor[Bool], ()>, <Func, NoShape>, <Func, NoShape>) -> (<Func, NoShape>)
 68       # Fullname with scope: (Default/Switch-op4)
 69       # In file t6.py:15~20, 4~16/    def construct(self, x, y):/
 70       # In file t6.py:18~19, 8~43/        if b :/
 71   %6(CNode_306) = %5[@FuncUnion(@20_4_✓__main___Net_construct_311, @21_14_✗__main___Net_construct_314)]()
 72       : () -> (<Tensor[Float32], ()>)
 73       # Fullname with scope: (5)
 74       # In file t6.py:15~20, 4~16/    def construct(self, x, y):/
 75       # In file t6.py:18~19, 8~43/        if b :/
 76   Return(%6) cnode_attrs: {checkpoint: Bool(1)}
 77       : (<Tensor[Float32], ()>)
 78       # Fullname with scope: (Default/Return-op19)
 79       # In file t6.py:15~20, 4~16/    def construct(self, x, y):/
 80       # In file t6.py:18~19, 8~43/        if b :/
 81 }
 82
 83
 84 indirect: 1
 85 subgraph attr:
 86 defer_inline: 0
 87 undeterminate: 0
 88 subgraph instance: 20_4_✓__main___Net_construct_311 : 0x135400a18
 89 # Parameters: 2, (<Tensor[Float32], ()>, <Tensor[Float32], ()>)
 90 # In file t6.py:15~20, 4~16/    def construct(self, x, y):/
 91 subgraph @20_4_✓__main___Net_construct_311(%para3_Parameter_320, %para4_Parameter_319) {
 92   %0(output) = PrimFunc_Div(%para4_Parameter_319, %para3_Parameter_320)
 93       : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
 94       # Fullname with scope: (Default/Div-op1)
 95       # In file t6.py:15~20, 4~16/    def construct(self, x, y):/
 96       # In file t6.py:19, 27~42/            b = ops.mul(b, self.func(a, b))/
 97       # In file t6.py:19, 27~36/            b = ops.mul(b, self.func(a, b))/<~~This line of code can be shared by multiple nodes, and may be duplicated./
 98       # In file t6.py:12~13, 4~28/    def func(x, y):/
 99       # In file t6.py:13, 15~28/        return ops.div(x, y)/
100       # In file t6.py:13, 15~22/        return ops.div(x, y)/<~~This line of code can be shared by multiple nodes, and may be duplicated./
101       # In file /workspace/mindspore/build/package/mindspore/ops/function/math_func.py:707~766, 0~17/def div(input, other, *, rounding_mode=None):/
102       # In file /workspace/mindspore/build/package/mindspore/ops/function/math_func.py:762~765, 4~38/    if rounding_mode:/
103       # In file /workspace/mindspore/build/package/mindspore/ops/function/math_func.py:765, 17~38/        output = P.Div()(input, other)/<~~This line of code can be shared by multiple nodes, and may be duplicated./
104   %1(CNode_313$b) = PrimFunc_Mul(%para3_Parameter_320, %0) cnode_attrs: {is_dynamic_len: Bool(0)}
105       : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
106       # Fullname with scope: (Default/Mul-op1)
107       # In file t6.py:15~20, 4~16/    def construct(self, x, y):/
108       # In file t6.py:19, 16~43/            b = ops.mul(b, self.func(a, b))/
109       # In file t6.py:19, 16~23/            b = ops.mul(b, self.func(a, b))/<~~This line of code can be shared by multiple nodes, and may be duplicated./
110       # In file /workspace/mindspore/build/package/mindspore/ops/auto_generate/gen_ops_def.py:3471~3518, 0~31/def mul(input, other):/
111       # In file /workspace/mindspore/build/package/mindspore/ops/auto_generate/gen_ops_def.py:3518, 11~31/    return mul_op(input, other)/
112   Return(%1)
113       : (<Tensor[Float32], ()>)
114       # Fullname with scope: (Default/Return-op20)
115       # In file t6.py:15~20, 4~16/    def construct(self, x, y):/
116       # In file t6.py:19, 12~43/            b = ops.mul(b, self.func(a, b))/
117 }
118
119
120 indirect: 1
121 subgraph attr:
122 defer_inline: 0
123 undeterminate: 0
124 subgraph instance: 21_14_✗__main___Net_construct_314 : 0x1353ff218
125 # Parameters: 1, (<Tensor[Float32], ()>)
126 # In file t6.py:15~20, 4~16/    def construct(self, x, y):/
127 subgraph @21_14_✗__main___Net_construct_314(%para5_Parameter_322) {
128   Return(%para5_Parameter_322)
129       : (<Tensor[Float32], ()>)
130       # Fullname with scope: (Default/Return-op21)
131       # In file t6.py:15~20, 4~16/    def construct(self, x, y):/
132       # In file t6.py:18~19, 8~43/        if b :/
133 }

```

以上内容可分为两个部分，第一部分为图的输入信息，第二部分为图的结构信息：

- 第1行表示该网络的顶图名称 `@19_1___main___Net_construct_304`，也就是入口图。
- 第2行表示该网络解析出来的图的数量，该IR文件展示了三张图的信息。分别为第23行的入口图`@19_1___main___Net_construct_304`；第84行的图`20_4_✓__main___Net_construct_311`，对应着网络中if条件为true时所运行的图；第120行的图`21_14_✗__main___Net_construct_314`，即对应着网络中if条件为false时所运行的图。
- 第14行表示该网络有多少个输入。
- 第16-17行是输入列表，遵循`%para[序号]_[name] : <[data_type], (shape)>`的格式。

对于具体的图来说（此处以图`@19_1___main___Net_construct_304`为例）：

- 第23-81行展示了图结构的信息，图中含有若干个节点，即`CNode`。该图包含`Sub`、`Add`、`Mul`这些在网路所调用的接口中所用到的算子。

`CNode`（[ANF-IR的设计请查看](https://www.mindspore.cn/docs/zh-CN/master/design/all_scenarios.html#文法定义)）的信息遵循如下格式，从左到右分别为序号、节点名称-debug_name、算子名称-op_name、输入节点-arg、节点的属性-primitive_attrs、输入和输出的规格、源码解析调用栈等信息。
由于ANF图为单向无环图，所以此处仅根据输入关系来体现节点与节点的连接关系。关联代码行则体现了`CNode`与脚本源码之间的关系，例如第75行表明该节点是由脚本中`if b`这一行解析而来。

```text
%[序号]([debug_name]) = [op_name]([arg], ...) primitive_attrs: {[key]: [value], ...}
    : (<[输入data_type]x[输入shape]>, ...) -> (<[输出data_type]x[输出shape]>, ...)
    # 关联代码行
```

关于关联代码行的说明：

- 代码行信息包括文件路径，代码起始位置和终止位置。例如：`# In file /workspace/mindspore/build/package/mindspore/nn/wrap/cell_wrapper.py:437~441, 8~45`，则文件路径为：`/workspace/mindspore/build/package/mindspore/nn/wrap/cell_wrapper.py`，代码起始位置为：第437行/第8列，代码终止位置为：第441行/第45列。如果代码不跨行，则不显示终止行信息，例如：`# In file /workspace/mindspore/build/package/mindspore/nn/wrap/cell_wrapper.py:418, 19~37`，只显示第418行。
- 代码行展示有两种模式，第一种是显示完整的调用栈，第二种为了减小文件的体积，只显示第一行，即省去了调用过程。默认按第一种模式，在所有ir文件中展示完整调用栈的代码行信息。
- 如果算子是反向传播算子，关联代码行除了会显示本身的代码，还会显示对应的正向代码，通过“Corresponding forward node candidate:”标识。
- 如果算子是融合算子，关联代码行会显示出融合的相关代码，通过“Corresponding code candidate:”标识，其中用分隔符“-”区分不同的代码。

> - 经过编译器的若干优化处理后，节点可能经过了若干转换（如算子拆分、算子融合等），节点的源码解析调用栈信息与脚本可能无法完全一一对应，这里仅作为辅助手段。
> - 在后端经过算子选择阶段后，输入输出规格信息（即`:`后内容）会有两行。第一行表示为`HOST`侧的规格信息，第二行为`DEVICE`侧的规格信息。

### dot文件介绍

可以用[graphviz](http://graphviz.org)将`dot`格式的IR文件作为输入生成图片。例如，在Linux操作系统下，可以通过以下命令转换成一张PNG图片。

```shell
dot -Tpng -o 01_type_inference_0003.png 01_type_inference_0003.dot
```

转换之后得到类似下图的模型示意图，可以观察构建的静态图模型结构。不同的黑框区分了不同的子图，图与图之间的蓝色箭头表示相互之间的调用。蓝色区域表示参数，矩形表示图的参数列表，六边形和黑色箭头表示该参数作为CNode的输入参与计算过程。黄色矩形表示CNode节点，从图中可以看出，CNode输入从下标0开始，第0个输入（即紫色或绿色区域）表示该算子将要进行怎样的计算，通过虚箭头连接。类型一般为算子原语，也可以是另一张图。下标1之后的输入则为计算所需要的参数。

![01_type_inference_0003.png](./images/dot_to_png.png)

## 如何根据analyze_fail.ir文件分析图推导失败的原因

MindSpore在编译图的过程中，经常会出现`type_inference`阶段的图推导失败的报错，开发者通常可以根据报错信息以及analyze_fail.ir文件，来定位出脚本中存在的问题。

### 例子1：参数数量不匹配

```python
  1 import os
  2 import mindspore as ms
  3 import mindspore.nn as nn
  4 from mindspore import ops
  5
  6 ms.set_context(mode=ms.GRAPH_MODE)
  7 os.environ['MS_DEV_SAVE_GRAPHS'] = '2'
  8 os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = './ir'
  9
 10 class Net(nn.Cell):
 11     def __init__(self):
 12         super().__init__()
 13
 14     def func(x, y):
 15         return ops.div(x, y)
 16
 17     def construct(self, x, y):
 18         a = ops.sub(x, 1)
 19         b = ops.add(a, y)
 20         c = ops.mul(b, self.func(a, a, b))
 21
 22 input1 = ms.Tensor(3, ms.float32)
 23 input2 = ms.Tensor(2, ms.float32)
 24 net = Net()
 25 out = net(input1, input2)
 26 print(out)
```

会出现如下的报错：

```text
  1 Traceback (most recent call last):
  2   File "t2.py", line 23, in <module>
  3     out = net(input1, input2)
  4   File "/workspace/mindspore/build/package/mindspore/nn/cell.py", line 701, in __call__
  5     out = self.compile_and_run(*args, **kwargs)
  6   File "/workspace/mindspore/build/package/mindspore/nn/cell.py", line 1051, in compile_and_run
  7     self.compile(*args, **kwargs)
  8   File "/workspace/mindspore/build/package/mindspore/nn/cell.py", line 1034, in compile
  9     _cell_graph_executor.compile(self, *self._compile_args, phase=self.phase,
 10   File "/workspace/mindspore/build/package/mindspore/common/api.py", line 1815, in compile
 11     result = self._graph_executor.compile(obj, args, kwargs, phase, self._use_vm_mode())
 12 TypeError: The parameters number of the function is 2, but the number of provided arguments is 3.
 13 FunctionGraph ID : func_40
 14 NodeInfo: In file t2.py:12~13, 4~28
 15     def func(x, y):
 16
 17 ----------------------------------------------------
 18 - C++ Call Stack: (For framework developers)
 19 ----------------------------------------------------
 20 mindspore/ccsrc/pipeline/jit/ps/static_analysis/stack_frame.cc:104 DoJump
 21
 22 ----------------------------------------------------
 23 - The Traceback of Net Construct Code:
 24 ----------------------------------------------------
 25 # 0 In file t2.py:18, 23~41
 26         c = ops.mul(b, self.func(a, a, b))
 27                        ^~~~~~~~~~~~~~~~~~
 28  (See file '/workspace/mindspore/rank_0/om/analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir)
```

以上的报错信息为：“TypeError: The parameters number of the function is 2, but the number of provided arguments is 3...”。
表明`FunctionGraph ID : func_40`只需要2个参数，但是却提供了3个参数。从“The function call stack ...”中，可以知道出错的代码为：“In file t2.py:18 ... self.func(a, a, b)”，是因为该处的函数调用传入参数的数目过多。

但如果报错信息不直观或者需要查看IR中已推导出的部分图信息，使用文本编辑软件（例如，vi）打开报错信息中的提示的文件（第28行括号中）：`/workspace/mindspore/rank_0/om/analyze_fail.ir`，文件中除了上述报错信息，还有如下内容（此处版本为MindSpore 2.3，后续版本中内容可能会有一些细微变化）：

```text
  1 # ===============================================================================================
  2 # The following shows the IR when the function graphs evaluation fails to help locate the problem.
  3 # You can search the last ------------------------> to the node which is evaluated failure.
  4 # Refer to https://www.mindspore.cn/search?inputValue=analyze_fail.ir to get more instructions.
  5 # ===============================================================================================
  6
  7 # IR entry: @__main___Net_construct_11
  8 # Total subgraphs: 0
  9
 10 # Total params: 2
 11 # Params:
 12 %para1_x: <null>
 13 %para2_y: <null>
 14
 15 subgraph attr:
 16 subgraph instance: __main___Net_construct_11 : 0x12fb85c18
 17 # In file t2.py:15~18, 4~42/    def construct(self, x, y):/
 18 subgraph @__main___Net_construct_11() {
 19   %0(CNode_3) = resolve(NameSpace[Entry: '__main__.Net.construct'], __main__.Net.construct)
 20       : (<External, NoShape>, <External, NoShape>) -> (<Func, NoShape>)
 21       #scope: (Default)
 22
 23 #------------------------> 0
 24   %1(CNode_2) = %0(%para1_x, %para2_y)
 25       : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<null>)
 26       #scope: (Default)
 27   Return(%1)
 28       : (<null>)
 29       #scope: (Default)
 30       # In file t2.py:15~18, 4~42/    def construct(self, x, y):/
 31 }
 32 # Order:
 33 #   1: @__main___Net_construct_11:CNode_3{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> Entry: '__main__.Net.construct', [2]: ValueNode<Symbol> __main__.Net.construct}
 34 #   2: @__main___Net_construct_11:CNode_2{[0]: CNode_3, [1]: param_x, [2]: param_y}
 35 #   3: @__main___Net_construct_11:CNode_44{[0]: ValueNode<Primitive> Return, [1]: CNode_2}
 36
 37
 38 subgraph attr:
 39 subgraph instance: __main___Net_construct_11 : 0x12fac0218
 40 # In file t2.py:15~18, 4~42/    def construct(self, x, y):/
 41 subgraph @__main___Net_construct_11(%para0_x, %para0_y) {
 42   %0(CNode_12) = resolve(NameSpace[SymbolStr: 'Namespace:__main__'], ops)
 43       : (<External, NoShape>, <External, NoShape>) -> (<External, NoShape>)
 44       #scope: (Default)
 45       # In file t2.py:16, 12~15/        a = ops.sub(x, 1)/
 46   %1(CNode_17) = getattr(%0, "mul")
 47       : (<External, NoShape>, <String, NoShape>) -> (<Func, NoShape>)
 48       #scope: (Default)
 49       # In file t2.py:18, 12~19/        c = ops.mul(b, self.func(a, a, b))/
 50   %2(CNode_15) = getattr(%0, "add")
 51       : (<External, NoShape>, <String, NoShape>) -> (<Func, NoShape>)
 52       #scope: (Default)
 53       # In file t2.py:17, 12~19/        b = ops.add(a, y)/
 54   %3(CNode_13) = getattr(%0, "sub")
 55       : (<External, NoShape>, <String, NoShape>) -> (<Func, NoShape>)
 56       #scope: (Default)
 57       # In file t2.py:16, 12~19/        a = ops.sub(x, 1)/
 58   %4(a) = %3(%para0_x, I64(1))
 59       : (<Tensor[Float32], ()>, <Int64, NoShape>) -> (<Tensor[Float32], ()>)
 60       #scope: (Default)
 61       # In file t2.py:16, 12~25/        a = ops.sub(x, 1)/
 62   %5(b) = %2(%4, %para0_y)
 63       : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
 64       #scope: (Default)
 65       # In file t2.py:17, 12~25/        b = ops.add(a, y)/
 66   %6(CNode_18) = resolve(NameSpace[ClassMember: 'Namespace:__main__..<Net::5265583376>'], func)
 67       : (<External, NoShape>, <External, NoShape>) -> (<Func, NoShape>)
 68       #scope: (Default)
 69       # In file t2.py:18, 23~32/        c = ops.mul(b, self.func(a, a, b))/
 70
 71 #------------------------> 1
 72   %7(CNode_19) = %6(%4, %4, %5)
 73       : (<Tensor[Float32], ()>, <Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<null>)
 74       #scope: (Default)
 75       # In file t2.py:18, 23~41/        c = ops.mul(b, self.func(a, a, b))/
 76   %8(c) = %1(%5, %7)
 77       : (<Tensor[Float32], ()>, <null>) -> (<null>)
 78       #scope: (Default)
 79       # In file t2.py:18, 12~42/        c = ops.mul(b, self.func(a, a, b))/
 80   %9(CNode_22) = StopGradient(%8)
 81       : (<null>) -> (<null>)
 82       #scope: (Default)
 83       # In file t2.py:15~18, 4~42/    def construct(self, x, y):/
 84   %10(CNode_21) = Depend(None, %9) primitive_attrs: {side_effect_propagate: I64(1)} cnode_attrs: {topo_sort_rhs_first: Bool(1)}
 85       : (<null>, <null>) -> (<null>)
 86       #scope: (Default)
 87       # In file t2.py:15~18, 4~42/    def construct(self, x, y):/
 88   Return(%10)
 89       : (<null>)
 90       #scope: (Default)
 91       # In file t2.py:15~18, 4~42/    def construct(self, x, y):/
 92 }
 93 # Order:
 94 #   1: @__main___Net_construct_11:CNode_12{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> SymbolStr: 'Namespace:__main__', [2]: ValueNode<Symbol> ops}
 95 #   2: @__main___Net_construct_11:CNode_13{[0]: ValueNode<Primitive> getattr, [1]: CNode_12, [2]: ValueNode<StringImm> sub}
 96 #   3: @__main___Net_construct_11:CNode_45{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> CommonOPS: 'Namespace:mindspore._extends.parse.trope', [2]: ValueNode<Symbol> MakeTuple}
 97 #   5: @__main___Net_construct_11:a{[0]: CNode_13, [1]: param_x, [2]: ValueNode<Int64Imm> 1}
 98 #   6: @__main___Net_construct_11:CNode_15{[0]: ValueNode<Primitive> getattr, [1]: CNode_12, [2]: ValueNode<StringImm> add}
 99 #   7: @__main___Net_construct_11:CNode_46{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> CommonOPS: 'Namespace:mindspore._extends.parse.trope', [2]: ValueNode<Symbol> MakeTuple}
100 #   9: @__main___Net_construct_11:b{[0]: CNode_15, [1]: a, [2]: param_y}
101 #  10: @__main___Net_construct_11:CNode_17{[0]: ValueNode<Primitive> getattr, [1]: CNode_12, [2]: ValueNode<StringImm> mul}
102 #  11: @__main___Net_construct_11:CNode_18{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> ClassMember: 'Namespace:__main__..<Net::5265583376>', [2]: ValueNode<Symbol> func}
103 #  12: @__main___Net_construct_11:CNode_47{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> CommonOPS: 'Namespace:mindspore._extends.parse.trope', [2]: ValueNode<Symbol> MakeTuple}
104 #  14: @__main___Net_construct_11:CNode_19{[0]: CNode_18, [1]: a, [2]: a, [3]: b}
105 #  15: @__main___Net_construct_11:CNode_48{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> CommonOPS: 'Namespace:mindspore._extends.parse.trope', [2]: ValueNode<Symbol> MakeTuple}
106 #  17: @__main___Net_construct_11:c{[0]: CNode_17, [1]: b, [2]: CNode_19}
107 #  19: @__main___Net_construct_11:CNode_44{[0]: ValueNode<Primitive> Return, [1]: CNode_21}
108
109
110 # ===============================================================================================
111 # The total of function graphs in evaluation stack: 2
112 # ===============================================================================================
113
114
115 # ===============================================================================================
116 # The rest function graphs are the following:
117 # ===============================================================================================
118 No more function graphs.
```

`analyze_fail.ir`文件与前文介绍过的ir文件格式一致，唯一有区别的地方在于`analyze_fail.ir`文件中会指出推导出错的节点所在的位置，即第71行的`------------------------> 1`。该箭头指向了推导出错的节点，为`%7(CNode_19) = %6(%4, %4, %5) ...`。
根据`(%4, %4, %5)`可知，该节点的输入参数有三个。从源码解析调用栈中可以知道实际该函数为`self.func`，在脚本中的定义为`def func(x, y):...`。
在函数定义中，只需要两个参数，故会在此处出现推导失败的报错，需要修改脚本中传入的参数个数以解决该问题。

### 例子2：BiasAdd输入之间shape不匹配

```python
  1 import numpy as np
  2 import mindspore as ms
  3 from mindspore import nn, ops, set_context, Tensor, Parameter
  4 from mindspore.common.initializer import initializer
  5
  6 ms.set_context(mode=ms.GRAPH_MODE)
  7
  8 class Net(nn.Cell):
  9     def __init__(self):
 10         super(Net, self).__init__()
 11         self.weight = Parameter(initializer('normal', [32, 8]), name="weight")
 12         self.bias = Parameter(initializer('zeros', [4]), name="bias")
 13
 14     def construct(self, x1):
 15         x = ops.matmul(x1, self.weight)
 16         x = ops.bias_add(x, self.bias)
 17         return x
 18
 19 net = Net()
 20 x = Tensor(np.arange(3*32).reshape(3, 32), ms.float32)
 21 out = net(x)
 22 print('out', out.shape)
```

会出现如下的报错：

```text
  1 Traceback (most recent call last):
  2   File "t2.py", line 21, in <module>
  3     out = net(x)
  4   File "/workspace/mindspore/build/package/mindspore/nn/cell.py", line 701, in __call__
  5     out = self.compile_and_run(*args, **kwargs)
  6   File "/workspace/mindspore/build/package/mindspore/nn/cell.py", line 1051, in compile_and_run
  7     self.compile(*args, **kwargs)
  8   File "/workspace/mindspore/build/package/mindspore/nn/cell.py", line 1034, in compile
  9     _cell_graph_executor.compile(self, *self._compile_args, phase=self.phase,
 10   File "/workspace/mindspore/build/package/mindspore/common/api.py", line 1815, in compile
 11     result = self._graph_executor.compile(obj, args, kwargs, phase, self._use_vm_mode())
 12 ValueError: For 'BiasAdd', bias[0] shape should be equal to input_x[1] shape when data_format is 0, but got bias shape: .[const vector]{4}, input_shape: [const vector]{3, 8}.
 13
 14 ----------------------------------------------------
 15 - C++ Call Stack: (For framework developers)
 16 ----------------------------------------------------
 17 mindspore/core/ops/ops_func_impl/bias_add.cc:71 CheckShapeValid
 18
 19 ----------------------------------------------------
 20 - The Traceback of Net Construct Code:
 21 ----------------------------------------------------
 22 # 0 In file t2.py:16, 11~37
 23        x = ops.bias_add(x, self.bias)
 24            ^~~~~~~~~~~~~~~~~~~~~~~~~~
 25 # 1 In file /workspace/mindspore/build/package/mindspore/ops/function/nn_func.py:6280, 11~37
 26     return bias_add_op(input_x, bias)
 27            ^~~~~~~~~~~~~~~~~~~~~~~~~~
 28  (See file '/workspace/mindspore/rank_0/om/analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir)
```

根据以上报错可知，是算子`BiasAdd`的第一个输入和第二个输入的`shape`不匹配导致的错误。为了进一步了解算子的`shape`是经过了什么样的变化，使用文本编辑软件（例如，vi）打开报错信息中的提示的文件：`/workspace/mindspore/rank_0/om/analyze_fail.ir`，文件中除了上述报错信息，还有如下内容（此处版本为MindSpore 2.3，后续版本中内容可能会有一些细微变化）：

```text
  1 # ===============================================================================================
  2 # The following shows the IR when the function graphs evaluation fails to help locate the problem.
  3 # You can search the last ------------------------> to the node which is evaluated failure.
  4 # Refer to https://www.mindspore.cn/search?inputValue=analyze_fail.ir to get more instructions.
  5 # ===============================================================================================
  6
  7 # IR entry: @__main___Net_construct_6
  8 # Total subgraphs: 0
  9
 10 # Total params: 3
 11 # Params:
 12 %para1_x1: <null>
 13 %para2_weight: <Ref[Tensor[Float32]], (32, 8), ref_key=weight>  :  has_default
 14 %para3_bias: <Ref[Tensor[Float32]], (4), ref_key=bias>  :  has_default
 15
 16 subgraph attr:
 17 subgraph instance: __main___Net_construct_6 : 0x128910818
 18 # In file t2.py:14~17, 4~15/    def construct(self, x1):/
 19 subgraph @__main___Net_construct_6() {
 20   %0(CNode_3) = resolve(NameSpace[Entry: '__main__.Net.construct'], __main__.Net.construct)
 21       : (<External, NoShape>, <External, NoShape>) -> (<Func, NoShape>)
 22       #scope: (Default)
 23
 24 #------------------------> 0
 25   %1(CNode_2) = %0(%para1_x1)
 26       : (<Tensor[Float32], (3, 32)>) -> (<null>)
 27       #scope: (Default)
 28   Return(%1)
 29       : (<null>)
 30       #scope: (Default)
 31       # In file t2.py:17, 7~15/       return x/
 32 }
 33 # Order:
 34 #   1: @__main___Net_construct_6:CNode_3{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> Entry: '__main__.Net.construct', [2]: ValueNode<Symbol> __main__.Net.construct}
 35 #   2: @__main___Net_construct_6:CNode_2{[0]: CNode_3, [1]: param_x1}
 36 #   3: @__main___Net_construct_6:CNode_162{[0]: ValueNode<Primitive> Return, [1]: CNode_2}
 37
 38
 39 subgraph attr:
 40 subgraph instance: __main___Net_construct_6 : 0x14bb64c18
 41 # In file t2.py:14~17, 4~15/    def construct(self, x1):/
 42 subgraph @__main___Net_construct_6(%para0_x1) {
 43   %0(CNode_7) = resolve(NameSpace[SymbolStr: 'Namespace:__main__'], ops)
 44       : (<External, NoShape>, <External, NoShape>) -> (<External, NoShape>)
 45       #scope: (Default)
 46       # In file t2.py:15, 11~14/       x = ops.matmul(x1, self.weight)/
 47   %1(CNode_11) = getattr(%0, "bias_add")
 48       : (<External, NoShape>, <String, NoShape>) -> (<Func, NoShape>)
 49       #scope: (Default)
 50       # In file t2.py:16, 11~23/       x = ops.bias_add(x, self.bias)/
 51   %2(CNode_8) = getattr(%0, "matmul")
 52       : (<External, NoShape>, <String, NoShape>) -> (<Func, NoShape>)
 53       #scope: (Default)
 54       # In file t2.py:15, 11~21/       x = ops.matmul(x1, self.weight)/
 55   %3(CNode_9) = resolve(NameSpace[ClassMember: 'Namespace:__main__..<Net::4300844784>'], weight)
 56       : (<External, NoShape>, <External, NoShape>) -> (<Ref[Tensor[Float32]], (32, 8)>)
 57       #scope: (Default)
 58       # In file t2.py:15, 26~37/       x = ops.matmul(x1, self.weight)/
 59   %4(x) = %2(%para0_x1, %3)
 60       : (<Tensor[Float32], (3, 32)>, <Ref[Tensor[Float32]], (32, 8)>) -> (<Tensor[Float32], (3, 8)>)
 61       #scope: (Default)
 62       # In file t2.py:15, 11~38/       x = ops.matmul(x1, self.weight)/
 63   %5(CNode_12) = resolve(NameSpace[ClassMember: 'Namespace:__main__..<Net::4300844784>'], bias)
 64       : (<External, NoShape>, <External, NoShape>) -> (<Ref[Tensor[Float32]], (4)>)
 65       #scope: (Default)
 66       # In file t2.py:16, 27~36/       x = ops.bias_add(x, self.bias)/
 67
 68 #------------------------> 1
 69   %6(x) = %1(%4, %5)
 70       : (<Tensor[Float32], (3, 8)>, <Ref[Tensor[Float32]], (4)>) -> (<null>)
 71       #scope: (Default)
 72       # In file t2.py:16, 11~37/       x = ops.bias_add(x, self.bias)/
 73   Return(%6)
 74       : (<null>)
 75       #scope: (Default)
 76       # In file t2.py:17, 7~15/       return x/
 77 }
 78 # Order:
 79 #   1: @__main___Net_construct_6:CNode_7{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> SymbolStr: 'Namespace:__main__', [2]: ValueNode<Symbol> ops}
 80 #   2: @__main___Net_construct_6:CNode_8{[0]: ValueNode<Primitive> getattr, [1]: CNode_7, [2]: ValueNode<StringImm> matmul}
 81 #   3: @__main___Net_construct_6:CNode_9{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> ClassMember: 'Namespace:__main__..<Net::4300844784>', [2]: ValueNode<Symbol> weight}
 82 #   4: @__main___Net_construct_6:CNode_163{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> CommonOPS: 'Namespace:mindspore._extends.parse.trope', [2]: ValueNode<Symbol> MakeTuple}
 83 #   6: @__main___Net_construct_6:x{[0]: CNode_8, [1]: param_x1, [2]: CNode_9}
 84 #   7: @__main___Net_construct_6:CNode_11{[0]: ValueNode<Primitive> getattr, [1]: CNode_7, [2]: ValueNode<StringImm> bias_add}
 85 #   8: @__main___Net_construct_6:CNode_12{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> ClassMember: 'Namespace:__main__..<Net::4300844784>', [2]: ValueNode<Symbol> bias}
 86 #   9: @__main___Net_construct_6:CNode_164{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> CommonOPS: 'Namespace:mindspore._extends.parse.trope', [2]: ValueNode<Symbol> MakeTuple}
 87 #  11: @__main___Net_construct_6:x{[0]: CNode_11, [1]: x, [2]: CNode_12}
 88 #  12: @__main___Net_construct_6:CNode_162{[0]: ValueNode<Primitive> Return, [1]: x}
 89
 90
 91 subgraph attr:
 92 subgraph instance: bias_add_14 : 0x128910e18
 93 # In file /workspace/mindspore/build/package/mindspore/ops/function/nn_func.py:6244~6280, 0~37/def bias_add(input_x, bias):/
 94 subgraph @bias_add_14(%para0_input_x, %para0_bias) {
 95   %0(CNode_15) = resolve(NameSpace[SymbolStr: 'Namespace:mindspore.ops.function.nn_func'], _get_cache_prim)
 96       : (<External, NoShape>, <External, NoShape>) -> (<Func, NoShape>)
 97       #scope: (Default)
 98       # In file /workspace/mindspore/build/package/mindspore/ops/function/nn_func.py:6279, 18~33/    bias_add_op = _get_cache_prim(P.BiasAdd)(data_format="NCHW")/
 99   %1(CNode_16) = resolve(NameSpace[SymbolStr: 'Namespace:mindspore.ops.function.nn_func'], P)
100       : (<External, NoShape>, <External, NoShape>) -> (<External, NoShape>)
101       #scope: (Default)
102       # In file /workspace/mindspore/build/package/mindspore/ops/function/nn_func.py:6279, 34~35/    bias_add_op = _get_cache_prim(P.BiasAdd)(data_format="NCHW")/
103   %2(CNode_17) = getattr(%1, "BiasAdd")
104       : (<External, NoShape>, <String, NoShape>) -> (<Func, NoShape>)
105       #scope: (Default)
106       # In file /workspace/mindspore/build/package/mindspore/ops/function/nn_func.py:6279, 34~43/    bias_add_op = _get_cache_prim(P.BiasAdd)(data_format="NCHW")/
107   %3(CNode_18) = %0(%2)
108       : (<Func, NoShape>) -> (<Func, NoShape>)
109       #scope: (Default)
110       # In file /workspace/mindspore/build/package/mindspore/ops/function/nn_func.py:6279, 18~44/    bias_add_op = _get_cache_prim(P.BiasAdd)(data_format="NCHW")/
111   %4(CNode_131) = resolve(NameSpace[CommonOPS: 'Namespace:mindspore._extends.parse.trope'], make_dict)
112       : (<External, NoShape>, <External, NoShape>) -> (<Func, NoShape>)
113       #scope: (Default)
114       # In file /workspace/mindspore/build/package/mindspore/ops/function/nn_func.py:6279, 18~64/    bias_add_op = _get_cache_prim(P.BiasAdd)(data_format="NCHW")/
115   %5(CNode_132) = resolve(NameSpace[CommonOPS: 'Namespace:mindspore._extends.parse.trope'], MakeTuple)
116       : (<External, NoShape>, <External, NoShape>) -> (<Func, NoShape>)
117       #scope: (Default)
118       # In file /workspace/mindspore/build/package/mindspore/ops/function/nn_func.py:6279, 18~64/    bias_add_op = _get_cache_prim(P.BiasAdd)(data_format="NCHW")/
119   %6(CNode_133) = %5("data_format")
120       : (<String, NoShape>) -> (<Tuple[String], TupleShape(NoShape)>)
121       #scope: (Default)
122       # In file /workspace/mindspore/build/package/mindspore/ops/function/nn_func.py:6279, 18~64/    bias_add_op = _get_cache_prim(P.BiasAdd)(data_format="NCHW")/
123   %7(CNode_135) = resolve(NameSpace[CommonOPS: 'Namespace:mindspore._extends.parse.trope'], MakeTuple)
124       : (<External, NoShape>, <External, NoShape>) -> (<Func, NoShape>)
125       #scope: (Default)
126       # In file /workspace/mindspore/build/package/mindspore/ops/function/nn_func.py:6279, 18~64/    bias_add_op = _get_cache_prim(P.BiasAdd)(data_format="NCHW")/
127   %8(CNode_136) = %7("NCHW")
128       : (<String, NoShape>) -> (<Tuple[String], TupleShape(NoShape)>)
129       #scope: (Default)
130       # In file /workspace/mindspore/build/package/mindspore/ops/function/nn_func.py:6279, 18~64/    bias_add_op = _get_cache_prim(P.BiasAdd)(data_format="NCHW")/
131   %9(CNode_21) = %4(%6, %8)
132       : (<Tuple[String], TupleShape(NoShape)>, <Tuple[String], TupleShape(NoShape)>) -> (<Dictionary[[data_format,],[String]], NoShape>)
133       #scope: (Default)
134       # In file /workspace/mindspore/build/package/mindspore/ops/function/nn_func.py:6279, 18~64/    bias_add_op = _get_cache_prim(P.BiasAdd)(data_format="NCHW")/
135   %10(bias_add_op) = UnpackCall_unpack_call(%3, %9)
136       : (<Func, NoShape>, <Dictionary[[data_format,],[String]], NoShape>) -> (<Func, NoShape>)
137       #scope: (Default)
138       # In file /workspace/mindspore/build/package/mindspore/ops/function/nn_func.py:6279, 18~64/    bias_add_op = _get_cache_prim(P.BiasAdd)(data_format="NCHW")/
139
140 #------------------------> 2
141   %11(CNode_22) = %10(%para0_input_x, %para0_bias)
142       : (<Tensor[Float32], (3, 8)>, <Ref[Tensor[Float32]], (4)>) -> (<null>)
143       #scope: (Default)
144       # In file /workspace/mindspore/build/package/mindspore/ops/function/nn_func.py:6280, 11~37/    return bias_add_op(input_x, bias)/
145   Return(%11)
146       : (<null>)
147       #scope: (Default)
148       # In file /workspace/mindspore/build/package/mindspore/ops/function/nn_func.py:6280, 4~37/    return bias_add_op(input_x, bias)/
149 }
150 # Order:
151 #   1: @bias_add_14:CNode_15{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> SymbolStr: 'Namespace:mindspore.ops.function.nn_func', [2]: ValueNode<Symbol> _get_cache_prim}
152 #   2: @bias_add_14:CNode_16{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> SymbolStr: 'Namespace:mindspore.ops.function.nn_func', [2]: ValueNode<Symbol> P}
153 #   3: @bias_add_14:CNode_17{[0]: ValueNode<Primitive> getattr, [1]: CNode_16, [2]: ValueNode<StringImm> BiasAdd}
154 #   4: @bias_add_14:CNode_166{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> CommonOPS: 'Namespace:mindspore._extends.parse.trope', [2]: ValueNode<Symbol> MakeTuple}
155 #   6: @bias_add_14:CNode_18{[0]: CNode_15, [1]: CNode_17}
156 #   7: @bias_add_14:CNode_132{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> CommonOPS: 'Namespace:mindspore._extends.parse.trope', [2]: ValueNode<Symbol> MakeTuple}
157 #   8: @bias_add_14:CNode_133{[0]: CNode_132, [1]: ValueNode<StringImm> data_format}
158 #   9: @bias_add_14:CNode_135{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> CommonOPS: 'Namespace:mindspore._extends.parse.trope', [2]: ValueNode<Symbol> MakeTuple}
159 #  10: @bias_add_14:CNode_136{[0]: CNode_135, [1]: ValueNode<StringImm> NCHW}
160 #  11: @bias_add_14:CNode_131{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> CommonOPS: 'Namespace:mindspore._extends.parse.trope', [2]: ValueNode<Symbol> make_dict}
161 #  12: @bias_add_14:CNode_21{[0]: CNode_131, [1]: CNode_133, [2]: CNode_136}
162 #  13: @bias_add_14:bias_add_op{[0]: ValueNode<UnpackCall> MetaFuncGraph-unpack_call.20, [1]: CNode_18, [2]: CNode_21}
163 #  14: @bias_add_14:CNode_167{[0]: ValueNode<Primitive> resolve, [1]: ValueNode<NameSpace> CommonOPS: 'Namespace:mindspore._extends.parse.trope', [2]: ValueNode<Symbol> MakeTuple}
164 #  16: @bias_add_14:CNode_22{[0]: bias_add_op, [1]: param_input_x, [2]: param_bias}
165 #  17: @bias_add_14:CNode_165{[0]: ValueNode<Primitive> Return, [1]: CNode_22}
166
167
168 # ===============================================================================================
169 # The total of function graphs in evaluation stack: 3/5 (Ignored 2 internal frames).
170 # ===============================================================================================
171
172
173 # ===============================================================================================
174 # The rest function graphs are the following:
175 # ===============================================================================================
176 No more function graphs.
```

搜索`------------------------>`来到第68行，即推导出错的位置。根据`...(%4, %5): (<Tensor[Float32], (3, 8)>, <Ref[Tensor[Float32]], (4)>) -> (`<null>`)`可知，算子`BiasAdd`的输入是`%4`和`%5`这两个节点。其中，`%4`的shape是`[3, 8]`，`%5`的shape是`[4]`，不符合算子API中`BiasAdd`算子的描述`bias (Tensor) - 偏置Tensor，shape为 (C)。C必须与 input_x 的通道维度C相同...`的要求，故此处报错。

因此，为了解决该问题，可以修改`%4`的shape，或修改`%5`（即`self.bias`）的shape。

- 如果修改`%5`（也就是`self.bias`）的维度，只需要改成`self.bias = Parameter(initializer('zeros', [8]), name="bias")`。
- 如果修改`%4`的shape，先要明白`%4`是什么。根据第59行可知，这是一个`MatMul`算子，输出shape是`[3, 8]`。该算子的输入是`(%para0_x1, %3)`，第一个输入的shape是`[3, 32]`（即传入的参数`x`），第二个输入shape是`[32, 8]`（即`self.weight`）。为了满足和shape为`[4]`的数据`BiasAdd`的要求，需要使得`%4`的输出shape为`[3, 4]`，因此修改`self.weight`为`self.weight = Parameter(initializer('normal', [32, 4]), name="weight")`。
