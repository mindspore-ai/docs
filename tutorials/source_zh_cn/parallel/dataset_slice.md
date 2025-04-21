# 数据集切分

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_zh_cn/parallel/dataset_slice.md)

## 简介

在进行分布式训练时，以图片数据为例，当单张图片的大小过大时，如遥感卫星等大幅面图片，当单张图片过大时，需要对图片进行切分，每张卡读取一部分图片，进行分布式训练。处理数据集切分的场景，需要配合模型并行一起才能达到预期的降低显存的效果，因此，基于自动并行提供了该项功能。本教程使用的样例不是大幅面的网络，仅作示例。真实应用到大幅面的网络时，往往需要详细设计并行策略。

> 数据集切分在数据并行模式下不涉及。

### 相关接口

1. `mindspore.dataset.vision.SlicePatches(num_height=1, num_width=1)`：在水平和垂直方向上将Tensor切片为多个块。适合于Tensor高宽较大的使用场景。其中`num_height`为垂直方向的切块数量，`num_width`为水平方向的切块数量。更多参数可以参考[SlicePatches](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset_vision/mindspore.dataset.vision.SlicePatches.html)。

2. `dataset_strategy(config=((1, 1, 1, 8), (8,)))`：表示数据集分片策略，具体可以参考[AutoParallel并行配置](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/parallel/mindspore.parallel.auto_parallel.AutoParallel.html)。`dataset_strategy`接口有以下几点限制：

    - 每个输入至多允许在一维进行切分。如支持`dataset_strategy(config=((1, 1, 1, 8), (8,)))`或者`config=((1, 1, 1, 8), (1,))`，每个输入至多切分了一维；但是不支持`config=((1, 1, 4, 2), (1,))`，其第一个输入切分了两维。

    - 维度最高的一个输入，切分的数目，一定要比其他维度的多。如支持`config=((1, 1, 1, 8), (8,))`或者`config=((1, 1, 1, 1), (1,))`，其维度最多的输入为第一个输入，切分份数为8，其余输入切分均不超过8；但是不支持`config=((1, 1, 1, 1), (8,))`，其维度最多的输入为第一维，切分份数为1，但是其第二个输入切分份数却为8，超过了第一个输入的切分份数。

## 操作实践

### 样例代码说明

> 下载完整的样例代码：[dataset_slice](https://gitee.com/mindspore/docs/tree/r2.6.0rc1/docs/sample_code/dataset_slice)。

目录结构如下：

```text
└─ sample_code
    ├─ dataset_slice
       ├── train.py
       └── run.sh
    ...
```

其中，`train.py`是定义网络结构和训练过程的脚本。`run.sh`是执行脚本。

### 配置分布式环境

通过init初始化通信。

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
init()
```

### 数据集加载

使用数据集切分时，需要同时调用数据集的`SlicePatches`接口去构造数据集，并且，为了保证各卡读入数据一致，需要对数据集固定随机数种子。

```python
import os
import mindspore.dataset as ds
from mindspore import nn

slice_h_num = 1
slice_w_num = 4

ds.config.set_seed(1000) # set dataset seed to make sure that all cards read the same data
def create_dataset(batch_size):
    dataset_path = os.getenv("DATA_PATH")
    dataset = ds.MnistDataset(dataset_path)
    image_transforms = [
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW()
    ]
    label_transform = ds.transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    # slice image
    slice_patchs_img_op = ds.vision.SlicePatches(slice_h_num, slice_w_num)
    img_cols = ['img' + str(x) for x in range(slice_h_num * slice_w_num)]
    dataset = dataset.map(operations=slice_patchs_img_op, input_columns="image", output_columns=img_cols)
    dataset = dataset.project([img_cols[get_rank() % (slice_h_num * slice_w_num)], "label"])
    dataset = dataset.batch(batch_size)
    return dataset

data_set = create_dataset(32)
```

### 网络定义

此处网络定义与单卡模型一致，并通过 `no_init_parameters` 接口延后初始化网络参数和优化器参数：

```python
from mindspore import nn
from mindspore.nn.utils import no_init_parameters

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Dense(28*28, 512)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Dense(512, 512)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

with no_init_parameters():
    net = Network()
    optimizer = nn.SGD(net.trainable_params(), 1e-2)
```

### 训练网络

在这一步，需要定义损失函数以及训练过程，通过顶层 `AutoParallel` 类包裹 grad_fn 实现并行设置，设置并行模式为半自动并行模式`semi_auto`。此外，配置数据集`dataset_strategy`切分策略`config`为((1, 1, 1, 4), (1,))，代表垂直方向不切分，水平方向切分4块。
本例采用函数式方式编写，这部分与单卡模型一致：

```python
from mindspore import nn
import mindspore as ms
from mindspore.parallel.auto_parallel import AutoParallel

loss_fn = nn.CrossEntropyLoss()

def forward_fn(data, target):
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

# 设置并行、数据切分策略
grad_fn = AutoParallel(grad_fn, parallel_mode="semi_auto")
grad_fn.dataset_strategy(config=((1, 1, slice_h_num, slice_w_num), (1,)))

for epoch in range(1):
    i = 0
    for image, label in data_set:
        (loss_value, _), grads = grad_fn(image, label)
        optimizer(grads)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_value))
        i += 1
```

### 运行单机8卡脚本

接下来通过命令调用对应的脚本，以`msrun`启动方式，8卡的分布式训练脚本为例，进行分布式训练：

```bash
bash run.sh
```

训练完后，日志文件保存到`log_output`目录下，关于Loss部分结果保存在`log_output/worker_*.log`中，示例如下：

```text
epoch: 0, step: 0, loss is 2.281521
epoch: 0, step: 10, loss is 2.185312
epoch: 0, step: 20, loss is 1.9531741
epoch: 0, step: 30, loss is 1.6952474
epoch: 0, step: 40, loss is 1.2967496
...
```
