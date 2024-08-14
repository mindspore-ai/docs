# 精度调优

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/acc_debug.md)

## 调优常见问题及解决办法

- 精度调试阶段，可能会遇到以下常见问题：
    - 第一个loss和标杆对不齐:
         说明网络正向和标杆对不齐，可固定网络输入，关闭shuffle等随机性，在网络某些关键节点保存输出为npy，再借助[TroubleShooter比较两组Tensor值(npy文件)是否相等](https://gitee.com/mindspore/toolkits/blob/master/troubleshooter/docs/migrator.md#%E5%BA%94%E7%94%A8%E5%9C%BA%E6%99%AF4%E6%AF%94%E8%BE%83%E4%B8%A4%E7%BB%84tensor%E5%80%BCnpy%E6%96%87%E4%BB%B6%E6%98%AF%E5%90%A6%E7%9B%B8%E7%AD%89)，定位到第一个不一致的位置，再进行二分定位，分析正向哪里差异导致loss和标杆对不齐造成精度问题。
    - 第一个loss和标杆对齐，后续loss对不齐：
         这个大概率是网络反向出现问题。可借助[TroubleShooter比对MindSpore与PyTorch的ckpt/pth](https://gitee.com/mindspore/toolkits/blob/master/troubleshooter/docs/migrator.md#%E5%BA%94%E7%94%A8%E5%9C%BA%E6%99%AF2%E6%AF%94%E5%AF%B9mindspore%E4%B8%8Epytorch%E7%9A%84ckptpth)通过比较ckpt与pth的对应参数的值来检验网络反向更新的结果。
    - loss出现NAN/INF：
         可以通过[TroubleShooter获取INF/NAN值抛出点](https://gitee.com/mindspore/toolkits/blob/master/troubleshooter/docs/tracker.md#%E5%BA%94%E7%94%A8%E5%9C%BA%E6%99%AF2%E8%8E%B7%E5%8F%96infnan%E5%80%BC%E6%8A%9B%E5%87%BA%E7%82%B9)识别网络中第一个出现NAN或INF的位置。
         也可通过[Dump](https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/dump.html)工具进行溢出算子检测。

## 精度调试过程

精度调试的过程基本可以分为以下过程：

### 1.检查参数

这部分包含检查所有参数和可训练参数的数量，检查所有参数的shape。

- PyTorch可训练的参数用`Parameter`，不可训练的参数`Parameter`的`requires_grad=False`或使用`buffer`。
- MindSpore可训练的参数和不可训练的参数都用`Parameter`。
- MindSpore和PyTorch的参数除了BatchNorm区别大一点，其他都差不多。注意MindSpore里没有`num_batches_tracked`的对应，实际使用时这个参数可以用优化器里的`global_step`替代。

  | MindSpore | PyTorch |
  | --------- | --------|
  | gamma | weight |
  | beta | bias |
  | moving_mean | running_mean |
  | moving_variance | running_var |
  | 无 | num_batches_tracked |

<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch 获取参数方法 </td> <td style="text-align:center"> MindSpore 获取参数方法 </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
from torch import nn

class ptNet(nn.Module):
    def __init__(self):
          super(ptNet, self).__init__()
          self.fc = nn.Linear(1, 1)
    def construct(self, x):
        output = self.fc(x)
        return output

ptnet = ptNet()
all_parameter = []
trainable_params = []
# 获取网络里的参数
for name, item in ptnet.named_parameters():
    if item.requires_grad:
        trainable_params.append(item)
    all_parameter.append(item)
    print(name, item.shape)

for name, buffer in ptnet.named_buffers():
    all_parameter.append(buffer)
    print(name, buffer.shape)
print(f"all parameter numbers: {len(all_parameter)}")
print(f"trainable parameter numbers: {len(trainable_params)}")
```

打印结果：

```text
fc.weight torch.Size([1, 1])
fc.bias torch.Size([1])
all parameter numbers: 2
trainable parameter numbers: 2
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
from mindspore import nn

class msNet(nn.Cell):
    def __init__(self):
        super(msNet, self).__init__()
        self.fc = nn.Dense(1, 1, weight_init='normal')
    def construct(self, x):
        output = self.fc(x)
        return output

msnet = msNet()
# 获取所有参数
all_parameter = []
for item in msnet.get_parameters():
    all_parameter.append(item)
    print(item.name, item.data.shape)
print(f"all parameter numbers: {len(all_parameter)}")

# 获取可训练的参数
trainable_params = msnet.trainable_params()
for item in trainable_params:
    print(item.name, item.data.shape)
print(f"trainable parameter numbers: {len(trainable_params)}")
```

打印结果：

```text
fc.weight (1, 1)
fc.bias (1,)
all parameter numbers: 2
fc.weight (1, 1)
fc.bias (1,)
trainable parameter numbers: 2
```

</pre>
</td>
</tr>
</table>

### 2.模型验证

由于模型算法的实现是和框架没有关系的，训练好的参数可以先转换成MindSpore的[checkpoint](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/save_load.html)文件加载到网络中进行推理验证。

整个模型验证的流程请参考[resnet网络迁移](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/sample_code.html#%E6%A8%A1%E5%9E%8B%E9%AA%8C%E8%AF%81)。

### 3.推理验证

确认模型结构完全一致后，最好再做一次推理验证。整个推理过程除了模型外还有数据集和metrics，当推理结果不一致时，可以采用控制变量法，逐步排除问题。

整个推理验证的流程请参考[resnet网络迁移](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/sample_code.html#%E6%8E%A8%E7%90%86%E6%B5%81%E7%A8%8B)。

### 4.训练精度

当完成了推理验证后，我们基本可以确定基础模型，数据处理和metrics计算没有问题。此时如果训练的精度还是有问题时怎么进行排查呢？

- 加loss scale，在Ascend上因为Conv、Sort、TopK等算子只能是float16的，MatMul由于性能问题最好也是float16的，所以建议Loss scale操作作为网络训练的标配。

  Ascend 上只支持float16的算子列表：

  | 算子类别 | 具体算子 |
  | ------ | ------ |
  | 池化 | AdaptiveMaxPool2D, AvgPool3D, AvgPool, MaxPool, MaxPoolWithArgmax, Pooling |
  | 循环神经结构 | LSTM, DynamicRNN, GRUV2 |
  | 卷积 | Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose, DepthwiseConv2dNative |
  | 矩阵乘 (这类主要float32太慢, 需要转成float16的) | MatMul, BatchMatMul |
  | 排序 | Sort, TopK |
  | 其他 | BoundingBoxEncode, ExtractImagePatches, ExtractVolumePatches, FusedDbnDw, IOU, NewIm2Col, NMSWithMask |

  ```python
  import mindspore as ms
  from mindspore import nn
  from mindspore.train import Model
  # Model
  loss_scale_manager = ms.amp.FixedLossScaleManager(drop_overflow_update=False) # 静态loss scale
  # loss_scale_manager = ms.amp.DynamicLossScaleManager()   # 动态loss scale

  # 1. 一般流程
  loss = nn.MSELoss()
  opt = nn.Adam(params=msnet.trainable_params(), learning_rate=0.01)
  model = Model(network=msnet, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale_manager)

  # 2. 自已包装正向网络和loss函数
  msnet.to_float(ms.float16)
  loss.to_float(ms.float32)
  net_with_loss = nn.WithLossCell(msnet, loss)
  # 用Model的混合精度最好要有loss_fn，否则loss部分会使用float16计算，容易溢出
  model = Model(network=net_with_loss, optimizer=opt)

  # 3. 自己包装训练流程
  scale_sense = nn.FixedLossScaleUpdateCell(1)#(config.loss_scale) # 静态loss scale
  # scale_sense = nn.DynamicLossScaleUpdateCell(loss_scale_value=config.loss_scale,
  #                                             scale_factor=2, scale_window=1000) # 动态loss scale
  train_net = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer=opt, scale_sense=scale_sense)
  model = Model(network=train_net)
  ```

- 排查是否溢出，添加loss scale时，默认会加上溢出检测，可以将是否溢出的结果进行监测，如果持续溢出的话建议优先排查为什么溢出，建议使用MindSpore Insight的[dump数据](https://mindspore.cn/tutorials/experts/zh-CN/master/debug/dump.html)。

  ```python
  import numpy as np
  from mindspore import dataset as ds

  def get_data(num, w=2.0, b=3.0):
      for _ in range(num):
          x = np.random.uniform(-10.0, 10.0)
          noise = np.random.normal(0, 1)
          y = x * w + b + noise
          yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)

  def create_dataset(num_data, batch_size=16, repeat_size=1):
      input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
      input_data = input_data.batch(batch_size, drop_remainder=True)
      input_data = input_data.repeat(repeat_size)
      return input_data

  train_net.set_train()
  dataset = create_dataset(1600)
  iterator = dataset.create_tuple_iterator()
  for i, data in enumerate(iterator):
      loss, overflow, scaling_sens = train_net(*data)
      print("step: {}, loss: {}, overflow:{}, scale:{}".format(i, loss, overflow, scaling_sens))
  ```

  ```text
  step: 0, loss: 138.42825, overflow:False, scale:1.0
  step: 1, loss: 118.172104, overflow:False, scale:1.0
  step: 2, loss: 159.14542, overflow:False, scale:1.0
  step: 3, loss: 150.65671, overflow:False, scale:1.0
  ... ...
  step: 97, loss: 69.513245, overflow:False, scale:1.0
  step: 98, loss: 51.903114, overflow:False, scale:1.0
  step: 99, loss: 42.250656, overflow:False, scale:1.0
  ```

- 排查优化器、loss和参数初始化，整个训练过程除了模型、数据集外新加的部分只有优化器、loss和参数初始化，训练有问题时需要重点排查。尤其是loss和参数初始化，出现问题的概率较大。
- 多卡确认是否加seed保证多卡初始化一致，[自定义训练](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/model_development/training_and_evaluation.html#训练流程)确认是否进行梯度聚合。

  ```python
  import mindspore as ms
  ms.set_seed(1) # 会固定MindSpore的，numpy的，dataset的随机种子，API内部的需要在API属性设置
  ```

- 排查数据处理，通过可视化等方法查看数据处理是否符合预期，重点查看数据shuffle，是否有数据不匹配的情况。

更多精度调试策略请参考[精度调试](https://mindspore.cn/mindinsight/docs/zh-CN/master/accuracy_problem_preliminary_location.html)。
