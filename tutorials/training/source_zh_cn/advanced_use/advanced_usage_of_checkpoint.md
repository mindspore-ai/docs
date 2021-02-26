# checkpoint高级用法

- [基础用法](#基础用法)
- [高级用法](#高级用法)
    - [保存](#保存)
        - [手动保存ckpt](#手动保存ckpt)
        - [保存指定cell](#保存指定cell)
        - [异步保存ckpt](#异步保存ckpt)
        - [保存自定义参数字典](#保存自定义参数字典)
    - [加载](#加载)
        - [严格加载参数名](#严格加载参数名)
        - [过滤指定前缀](#过滤指定前缀)
    - [转化其他框架ckpt为ms格式](#转化其他框架ckpt为ms格式)

<!-- /TOC -->

## 基础用法

此文为MindSpore checkpoint的进阶用法，用于涉及使用checkpoint的复杂场景。

基础用法可参考：[保存加载参数](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/use/save_model.html#checkpoint)。

## 高级用法

### 保存

#### 手动保存ckpt

使用方法：`save_checkpoint`

应用场景：
1、保存网络的初始值
2、手动保存指定网络

```python
save_checkpoint(save_obj, ckpt_file_name)
net = LeNet()
save_checkpoint(net, 'resnet.ckpt')
```

#### 保存指定cell

使用方法：`ModelCheckpoin` 类的  `saved_network` 参数

应用场景：

- 只保存推理网的参数（不保存优化器的参数会使生成的ckpt文件大小减小一倍)
- 保存子网的参数，用于fune-tine任务

```python
net = LeNet5()
loss = SoftmaxCrossEntropyWithLogits()
opt = nn.Momentum()
model = Model(net, loss, opt)
config_ck = CheckpointConfig()
ckpoint = ModelCheckpoint(prefix="lenet";, config=config_ck, saved_network=net)
model.train(epoch_size, dataset, callbacks=ckpoint)
```

#### 异步保存ckpt

使用方法：`ModelCheckpoint` 类的 `async_save` 参数

应用场景：训练的模型参数量较大，可以边训练边保存，节省保存ckpt时写文件的时间

```python
ckpoint = ModelCheckpoint(prefix="lenet", config=config_ck, async_save=True)
model.train(epoch_size, dataset, callbacks=ckpoint)
```

#### 保存自定义参数字典

使用方法：构造一个`obj_dict` 传入 `save_checkpoint` 方法

使用场景：

- 训练过程中需要额外保存参数(lr、epoch_size等)为checkpoint文件
- 修改checkpoint里面的参数值后重新保存
- 把pytorch、tf的checkpoint文件转化为MindSpore的checkpoint文件

根据具体场景分为两种情况：

1、已有checkpoint文件，修改内容后重新保存

```python
param_list = load_checkpoint("resnet.ckpt")
# eg: param_list = [{"name": param_name, "data": param_data},...]
# del element
del param_list[2]
# add element "epoch_size"
param = {"name": "epoch_size"}
param["data"] = Tensor(10)
param_list.append(param)
# modify element
param_list[3]["data"] = Tensor(66)
# save a new checkpoint file
save_checkpoint(param_list, 'modify.ckpt')
```

2、自定义参数列表保存成checkpoint文件

```python
param_list = []
# save epoch_size
param = {"name": "epoch_size"}
param["data"] = Tensor(10)
param_list.append(param)
# save learning rate
param = {"name": "learning_rate"}
param["data"] = Tensor(0.01)
param_list.append(param)
# save a new checkpoint file
save_checkpoint(param_list, 'hyperparameters.ckpt')
```

### 加载

#### 严格匹配参数名

背景：ckpt文件中的权重参数到net中的时候，会优先匹配net和ckpt中name相同的parameter。匹配完成后，发现net中存在没有加载的parameter，会匹配net中后缀名称与ckpt相同的parameter.(原因是ms的parameter命名机制)

例如：会把ckpt中名为：`conv.0.weight`的参数值加载到net中名为`net.conv.0.weight`的parameter中。

如果想取消这种模糊匹配，只采取严格匹配机制，可以通过方法`load_param_into_net`中的`strict_load`参数控制，默认为False，表示采取模糊匹配机制。

```python
net = Lenet()
param_list = load_checkpoint("resnet.ckpt")
load_param_into_net(net, param_list, strict_load=True):
```

#### 过滤指定前缀

使用方法：`load_checkpoint`的`filter_prefix` 参数

使用场景：加载ckpt时，想要过滤某些包含特定前缀的parameter

- 加载ckpt时，不加载优化器中的parameter(eg：filter_prefix=’opt‘)
- 不加载卷积层的parameter(eg：filter_prefix=’conv‘)

```python
net = Lenet()
param_list = load_checkpoint("resnet.ckpt")
load_param_into_net(net, param_list, filter_prefix='opt'):
```

> 以上为使用MindSpore checkpoint功能的进阶用法，上述所有用法均可共同使用。

### 转化其他框架ckpt为ms格式

把其他框架的checkpoint文件转化成MindSpore格式

思路：不管是什么框架，checkpoint文件中保存的就是参数名和参数值，调用相应框架的读取接口后，获取到参数名和数值后，按照MindSpore格式，构建出对象，就可以直接调用ms接口保存成ms格式的checkpoint文件了。

其中主要的工作量为对比不同框架间的parameter 名称，做到两个框架的网络中所有parameter name一一对应(可以使用一个map进行映射)，下面代码的逻辑转化parameter格式，不包括对应parameter name。

```python
import torch
from mindspore import Tensor save_checkpoint

def pytorch2mindspore('torch_resnet.pth'):
    # read pth file
    par_dict = torch.load('torch_resnet.pth')['state_dict']
    params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        params_list.append(param_dict)
    save_checkpoint(params_list,  'ms_resnet.ckpt')
```
