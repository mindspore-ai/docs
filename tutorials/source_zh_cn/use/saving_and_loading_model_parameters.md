# 模型参数的保存和加载

<a href="https://gitee.com/mindspore/docs/blob/r0.3/tutorials/source_zh_cn/use/saving_and_loading_model_parameters.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

在模型训练过程中，可以添加检查点(CheckPoint)用于保存模型的参数，以便进行推理及中断后再训练使用。

使用场景如下：

- 训练后推理场景
    - 模型训练完毕后保存模型的参数，用于推理或预测操作。
    - 训练过程中，通过实时验证精度，把精度最高的模型参数保存下来，用于预测操作。
- 再训练场景
    - 进行长时间训练任务时，保存训练过程中的CheckPoint文件，防止任务异常退出后从初始状态开始训练。
    - Fine-tuning（微调）场景，即训练一个模型并保存参数，基于该模型，面向第二个类似任务进行模型训练。

MindSpore的CheckPoint文件是一个二进制文件，存储了所有训练参数的值。采用了Google的Protocol Buffers机制，与开发语言、平台无关，具有良好的可扩展性。
CheckPoint的protocol格式定义在`mindspore/ccsrc/utils/checkpoint.proto`中。

以下通过一个示例来介绍MindSpore保存和加载的功能，网络选取ResNet-50，数据集为MNIST。

## 模型参数保存
在模型训练的过程中，使用callback机制传入回调函数`ModelCheckpoint`对象，可以保存模型参数，生成CheckPoint文件。
通过`CheckpointConfig`对象可以设置CheckPoint的保存策略。
保存的参数分为网络参数和优化器参数。

`ModelCheckpoint()`提供默认配置策略，方便用户快速上手。
具体用法如下：
```python
from mindspore.train.callback import ModelCheckpoint
ckpoint_cb = ModelCheckpoint()
model.train(epoch_num, dataset, callbacks=ckpoint_cb)
```

用户可以根据具体需求对CheckPoint策略进行配置。
具体用法如下：

```python
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10)
ckpoint_cb = ModelCheckpoint(prefix='resnet50', directory=None, config=config_ck)
model.train(epoch_num, dataset, callbacks=ckpoint_cb)
```

上述代码中，首先需要初始化一个`CheckpointConfig`类对象，用来设置保存策略。
`save_checkpoint_steps`表示每隔多少个step保存一次，`keep_checkpoint_max`表示最多保留CheckPoint文件的数量。
`prefix`表示生成CheckPoint文件的前缀名，`directory`表示存放文件的目录。
创建一个`ModelCheckpoint`对象把它传递给model.train方法，就可以在训练过程中使用CheckPoint功能了。

生成的CheckPoint文件如下：

> - resnet50-graph.meta # 编译后的计算图
> - resnet50-1_32.ckpt  # CheckPoint文件后缀名为'.ckpt'
> - resnet50-2_32.ckpt  # 文件的命名方式表示保存参数所在的epoch和step数
> - resnet50-3_32.ckpt  # 表示保存的是第3个epoch的第32个step的模型参数
> - ...


如果用户使用相同的前缀名，运行多次训练脚本，可能会生成同名CheckPoint文件。
MindSpore为方便用户区分每次生成的文件，会在用户定义的前缀后添加"_"和数字加以区分。

例：`resnet50_3-2_32.ckpt` 表示运行第3次脚本生成的第2个epoch的第32个step的CheckPoint文件。

> 当保存的单个模型参数较大时(超过64M)，会因为Protobuf自身对数据大小的限制，导致保存失败。这时可通过设置环境变量`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`解除限制。


### CheckPoint配置策略

MindSpore提供了两种保存CheckPoint策略: 迭代策略和时间策略，可以通过创建`CheckpointConfig`对象设置相应策略。
`CheckpointConfig`中共有四个参数可以设置：

- save_checkpoint_steps: 表示每隔多少个step保存一个CheckPoint文件，默认值为1。
- save_checkpoint_seconds：表示每隔多少秒保存一个CheckPoint文件，默认值为0。
- keep_checkpoint_max：表示最多保存多少个CheckPoint文件，默认值为5。
- keep_checkpoint_per_n_minutes：表示每隔多少分钟保留一个CheckPoint文件，默认值为0。

`save_checkpoint_steps`和`keep_checkpoint_max`为迭代策略，根据训练迭代的次数进行配置。
`save_checkpoint_seconds`和`keep_checkpoint_per_n_minutes`为时间策略，根据训练的时长进行配置。

两种策略不能同时使用，迭代策略优先级高于时间策略，当同时设置时，只有迭代策略可以生效。
当参数显示设置为`None`时，表示放弃该策略。
在迭代策略脚本正常结束的情况下，会默认保存最后一个step的CheckPoint文件。


## 模型参数加载

保存好CheckPoint文件后，就可以对参数进行加载。

### 用于推理验证

针对仅推理场景可以使用`load_checkpoint`把参数直接加载到网络中，以便进行后续的推理验证。

示例代码如下：

```python
resnet = ResNet50()
load_checkpoint("resnet50-2_32.ckpt", net=resnet)
dateset_eval = create_dataset(os.path.join(mnist_path, "test"), 32, 1) # define the test dataset
loss = CrossEntropyLoss()
model = Model(resnet, loss)
acc = model.eval(dataset_eval)
```

`load_checkpoint`方法会把参数文件中的网络参数加载到模型中。加载后，网络中的参数就是CheckPoint保存的。
`eval`方法会验证训练后模型的精度。

### 用于再训练场景

针对任务中断再训练及fine-tuning场景，可以加载网络参数和优化器参数到模型中。

示例代码如下：
```python
# return a parameter dict for model
param_dict = load_checkpoint("resnet50-2_32.ckpt")
resnet = ResNet50()
opt = Momentum()
# load the parameter into net
load_param_into_net(resnet, param_dict)
# load the parameter into operator
load_param_into_net(opt, param_dict)
loss = SoftmaxCrossEntropyWithLogits()
model = Model(resnet, loss, opt)
model.train(epoch, dataset)
```

`load_checkpoint`方法会返回一个参数字典，`load_param_into_net`会把参数字典中相应的参数加载到网络或优化器中。

## 导出GEIR模型和ONNX模型
当有了CheckPoint文件后，如果想继续做推理，就需要根据网络和CheckPoint生成对应的模型，当前我们支持基于昇腾AI处理器的GEIR模型导出和基于GPU的通用ONNX模型的导出。
下面以GEIR为例说明模型导出的实现，代码如下：
```python
from mindspore.train.serialization import export
import numpy as np
resnet = ResNet50()
# return a parameter dict for model
param_dict = load_checkpoint("resnet50-2_32.ckpt")
# load the parameter into net
load_param_into_net(resnet, param_dict)
input = np.random.uniform(0.0, 1.0, size = [32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name = 'resnet50-2_32.pb', file_format = 'GEIR')
```
使用`export`接口之前，需要先导入`mindspore.train.serialization`。
`input`用来指定导出模型的输入shape以及数据类型。
如果要导出ONNX模型，只需要将`export`接口中的`file_format`参数指定为ONNX即可：`file_format = 'ONNX'`。
