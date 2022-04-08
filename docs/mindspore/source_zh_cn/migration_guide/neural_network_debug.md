# 网络调试

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/neural_network_debug.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

本章将介绍网络调试的基本思路、常用工具，以及一些常见问题处理。

## 网络调试的基本流程

网络调试的过程主要分为以下几个步骤：

1. 网络流程调试成功，网络执行整体不报错，正确输出loss值，且正常完成参数更新。

   一般情况下，使用`model.train`接口完整执行一个step并且不报错，即正常执行并完成了参数更新；如果需要精确确认，可以通过`mindspore.train.callback.CheckpointConfig`中的参数`save_checkpoint_steps=1`保存连续两个step的Checkpoint文件，或者使用`save_checkpoint`接口直接保存Checkpoint文件，然后通过以下代码打印Checkpoint文件中的权重值，查看两个step的权重是否发生改变，并完成更新。

   ```python
   import mindspore
   import numpy as np
   ckpt = mindspore.load_checkpoint(ckpt_path)
   for param in ckpt:
       value = ckpt[param].data.asnumpy()
       print(value)
   ```

2. 网络多轮迭代执行输出loss值，且loss值具有基本的收敛趋势。

3. 网络精度调试，超参调优。

## 网络调试中的常用方法

### 流程调试

本节内容介绍脚本开发基本完成后，网络流程调试过程中可能出现的问题和解决方法。

#### 用PyNative模式进行流程调试

在脚本开发和网络流程调试中，我们推荐使用PyNative模式进行调试。PyNative模式支持执行单算子、普通函数和网络，以及单独求梯度的操作。在PyNative模式下，可以方便地设置断点，获取网络执行的中间结果，也可以通过pdb的方式对网络进行调试。

在默认情况下，MindSpore处于Graph模式，可以通过`context.set_context(mode=context.PYNATIVE_MODE)`设置为PyNative模式，相关示例可参考[使用PyNative模式调试](https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/debug_in_pynative_mode.html#pynative)。

#### 获取更多报错信息

在网络流程调试过程中，如果需要获取更多的报错信息，可通过以下方式获得：

- 在PyNative模式下可使用pdb进行调试，利用pdb打印相关堆栈和上下文信息帮助问题定位。
- 使用Print算子打印更多上下文信息，具体示例可参考[Print算子功能介绍](https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/custom_debugging_info.html#print)。
- 调整日志级别获取更多报错信息，MindSpore可通过环境变量方便地调整日志级别，具体可参考[日志相关的环境变量和配置](https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/custom_debugging_info.html#日志相关的环境变量和配置)。

#### 常见错误

在网络流程调试中，常见的错误有以下几类：

- 算子执行报错

  网络流程调试过程中，常出现shape不匹配、dtype不支持等算子执行报错，此时应根据报错信息检查是否正确使用算子，以及算子输入数据的shape是否与预期相符，并进行相应修改。

  相关算子支持和API介绍可参考[算子支持列表](https://www.mindspore.cn/docs/zh-CN/master/note/operator_list.html)和[算子Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)。

- 相同脚本，在PyNative模式下能跑通，但Graph模式下报错

  MindSpore的Graph模式下，`construct`函数中的代码由MindSpore框架进行解析，有一些Python语法还未支持，因此导致报错。此时应当根据报错信息按照[MindSpore的语法说明](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html)修改相关代码。

- 分布式并行训练脚本配置错误

  分布式并行训练脚本及环境配置可参考[分布式并行训练教程](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/distributed_training.html)。

### loss值对比检查

在具有对标脚本的情况下，可对比对标脚本跑出的loss值与MindSpore脚本跑出的loss值，以验证整体网络结构和算子精度的正确性。

#### 主要步骤

1. 保证输入相同

    需保证两个网络中输入相同，使得在相同的网络结构下，网络输出相同。可使用以下几种方式保证输入相同：

    - 使用numpy自行构造输入数据，保证网络输入相同，MindSpore支持Tensor和numpy的自由转换。构造输入数据可以参考以下脚本：

      ```python
      input = Tensor(np.random.randint(0, 10, size=(3, 5, 10)).astype(np.float32))
      ```

    - 使用相同数据集进行计算，MindSpore支持使用TFRecord数据集，可使用`mindspore.dataset.TFRecordDataset`接口读取。

2. 去除网络中随机性因素的影响

   去除网络中的随机性影响，主要方法有设置相同的随机性种子，关掉数据shuffle，修改代码去除dropout、initializer等网络中具有随机性的算子的影响等。

3. 保证相关超参数的设置相同

   需保证网络中的超参数设置相同，以保证相同的输入，算子的输出相同。

4. 运行网络，比较输出的loss值，一般loss值误差在1‰左右，因为算子本身存在一定精度误差，随着step数增大，误差会有一定累加。

#### 相关问题定位

如果loss值误差较大，可使用以下几种方式进行问题定位：

- 检查输入、超参设置是否相同，以及是否完全去除了随机性影响。

  同一脚本多次重跑，loss值相差较大，则说明没有完全去除网络中的随机性影响。

- 整体判断。

  如果第一轮loss值就出现较大误差，则说明网络的前向计算就存在问题。

  如果第一轮loss值在误差范围内，第二轮开始loss值出现较大误差，则说明网络的前向计算应该没有问题，反向梯度和权重更新计算可能存在问题。

- 有了整体的判断之后，由粗到细进行输入输出数值的精度对比。

  首先，对各个子网从输入开始逐层对比输入输出值，确定初始出现问题的子网。

  然后，对比子网中的网络结构以及算子的输入输出，找到出现问题的网络结构或算子，进行修改。

  如果在此过程中发现了算子精度存在问题，可在[MindSpore代码托管平台](https://gitee.com/mindspore/mindspore)上提issue，相关人员将对问题进行跟踪处理。

- MindSpore提供了丰富的工具获取网络中间数据，可根据实际情况选用。

    - [数据Dump功能](https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/custom_debugging_info.html#dump)
    - [使用Print算子打印相关信息](https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/custom_debugging_info.html#print)
    - [使用可视化组件MindInsight](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/index.html)

### 精度调试工具

#### 自定义调试信息

- [Callback功能](https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/custom_debugging_info.html#callback)

   MindSpore已提供ModelCheckpoint、LossMonitor、SummaryCollector等Callback类用于保存模型参数、监控loss值、保存训练过程信息等功能，用户也可自定义Callback函数用于实现在每个epoch和step的开始和结束运行相关功能，具体示例可参考[自定义Callback](https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/custom_debugging_info.html#自定义callback)。

- [MindSpore metrics功能](https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/custom_debugging_info.html#mindspore-metrics)

   当训练结束后，可以使用metrics评估训练结果的好坏。MindSpore提供了多种metrics评估指标，如：`accuracy`、`loss`、`precision`、`recall`、`F1`等。

- [边训练边推理](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/evaluate_the_model_during_training.html)

   可通过定义推理的CallBack函数的方式在训练时进行推理。

- [自定义训练](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/train_and_eval.html)

- 自定义学习率

   MindSpore提供了一些常见的动态学习率实现以及一些常见的具有自适应学习率调整功能的优化器，可参考API文档中的[Dynamic Learning Rate](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.html#dynamic-learning-rate)和[Optimizer Functions](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.html#optimizer-functions)。

   同时，用户可实现自定义的动态学习率，以WarmUpLR为例：

   ```python
   class WarmUpLR(LearningRateSchedule):
       def __init__(self, learning_rate, warmup_steps):
           super(WarmUpLR, self).__init__()
           ## check the input
           if not isinstance(learning_rate, float):
               raise TypeError("learning_rate must be float.")
           validator.check_non_negative_float(learning_rate, "learning_rate", self.cls_name)
           validator.check_positive_int(warmup_steps, 'warmup_steps', self.cls_name)
           ## define the operators
           self.warmup_steps = warmup_steps
           self.learning_rate = learning_rate
           self.min = ops.Minimum()
           self.cast = ops.Cast()

       def construct(self, global_step):
           ## calculate the lr
           warmup_percent = self.cast(self.min(global_step, self.warmup_steps), mstype.float32)/ self.warmup_steps
           return self.learning_rate * warmup_percent
   ```

#### 使用MindOptimizer进行超参调优

MindSpore提供了MindOptimizer工具帮助用户进行更便捷的超参调优，详细示例和使用方法可参考[使用MindOptimizer进行超参调优](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/hyper_parameters_auto_tuning.html)。

#### loss值异常定位

对loss值为INF、NAN，或者loss值不收敛的情况，可从以下几种情况进行排查：

1. 检查loss_scale溢出。

   在混合精度使用loss_scale的场景下，出现loss值为INF、NAN的情况，可能是scale值过大造成的，如果为动态loss_scale，则会自动调整scale值；如果为静态loss_scale，则需要减小scale值。

   如果`scale=1`的情况下依旧存在loss值为INF、NAN的情况，则网络中应该有算子出现溢出，需要进行进一步定位。

2. 造成loss值异常的原因可能由输入数据异常、算子溢出、梯度消失、梯度爆炸等原因造成。

   排查算子溢出、梯度为0、权重异常、梯度消失和梯度爆炸等网络中间值情况，推荐使用[MindInsight调试器](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/debugger.html)设置相应检测点进行检测和调试，这种方式可较为全面地进行问题定位，可调试性也比较强。

   下面介绍几种简单的初步排查方法：

    - 观察权重值是否出现梯度爆炸的情况，也可通过加载保存的Checkpoint文件，打印权重值进行初步判断，打印权重值可参考以下代码：

        ```python
        import mindspore
        import numpy as np
        ckpt = mindspore.load_checkpoint(ckpt_path)
        for param in ckpt:
            value = ckpt[param].data.asnumpy()
            print(value)
        ```

    - 查看是否出现梯度为0的情况，也可以通过对比不同step保存的Checkpoint文件的权重值是否发生变化，进行初步判断，Checkpoint文件的权重值对比可参考以下代码：

        ```python
        import mindspore
        import numpy as np
        ckpt1 = mindspore.load_checkpoint(ckpt1_path)
        ckpt2 = mindspore.load_checkpoint(ckpt2_path)
        sum = 0
        same = 0
        for param1,param2 in zip(ckpt1,ckpt2):
            sum = sum + 1
            value1 = ckpt[param1].data.asnumpy()
            value2 = ckpt[param2].data.asnumpy()
            if value1 == value2:
                print('same value: ', param1, value1)
                same = same + 1
        print('All params num: ', sum)
        print('same params num: ', same)
        ```

    - 查看权重值中是否出现NAN、INF异常数据，也可通过加载Checkpoint文件进行简单判断，一般来说，权重值中出现NAN、INF，则梯度计算中也出现了NAN、INF，可能有溢出情况发生，相关代码可参考：

        ```python
        import mindspore
        import numpy as np
        ckpt = mindspore.load_checkpoint(ckpt_path)
        for param in ckpt:
            value = ckpt[param].data.asnumpy()
            if np.isnan(value):
                print('NAN value:', value)
            if np.isinf(value):
                print('INF value:', value)
        ```
