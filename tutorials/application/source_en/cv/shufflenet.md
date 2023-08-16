# ShuffleNet for Image Classification

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/application/source_en/cv/shufflenet.md)

> The current case does not support the static graph mode on the GPU device. Other modes are supported.

## ShuffleNet

ShuffleNetV1 is a computing-efficient CNN model proposed by Face++. Similar to MobileNet and SqueezeNet, it is mainly used on mobile devices. Therefore, the model is designed to use limited computing resources to achieve the best model accuracy. The core design of ShuffleNetV1 is to introduce two operations: pointwise group convolution and channel shuffle, which greatly reduce the calculation workload of the model while maintaining the accuracy. Similar to MobileNet, ShuffleNetV1 compresses and accelerates models by designing a more efficient network structure.

> For more details about ShuffleNet, see [ShuffleNet](https://arxiv.org/abs/1707.01083).

As shown in the following figure, ShuffleNet almost minimizes the number of parameters while maintaining the accuracy. Therefore, ShuffleNet has a fast calculation speed, and the number of parameters per unit contributes greatly to the model accuracy.

![shufflenet1](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/application/source_zh_cn/cv/images/shufflenet_1.png)

> Image source: Bianco S, Cadene R, Celona L, et al.Benchmark analysis of representative deep neural network architectures[J]. IEEE access, 2018, 6: 64270-64277.

## Model Architecture

The most prominent feature of ShuffleNet is that different channels are rearranged to solve the disadvantages of group convolution. By improving the Bottleneck unit of ResNet, high accuracy is achieved with a small amount of calculation.

### Pointwise Group Convolution

The following figure shows the principle of group convolution. Compared with common convolution, the size of the convolution kernel in each group is in_channels/g\*k\*k. There are *g* groups in total with (in_channels/g\*k\*k)\*out_channels parameters which is 1/g of the normal convolution parameters. In group convolution, each convolution kernel processes only some channels of the input feature map. **An advantage is that the quantity of parameters is reduced, but the quantity of output channels is still equal to that of convolution kernels.**

![shufflenet2](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/application/source_zh_cn/cv/images/shufflenet_2.png)

> Image source: Huang G, Liu S, Van der Maaten L, et al.Condensenet: An efficient densenet using learned group convolutions[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 2752-2761.

Depthwise convolution divides the *g* groups into `in_channels` that are equal to the number of input channels, and then performs a convolution operation on each `in_channels`. Each convolution kernel processes only one channel. The size of the convolution kernel is recorded as 1\*k\*k. In this case, the number of convolution kernel parameters is calculated as follows: in_channels\*k\*k. The number of obtained feature maps channels is the same as the number of input channels.

On the basis of group convolution, pointwise group convolution is assumed that the size of a convolution kernel of each group is $1\times 1$, and the quantity of convolution kernel parameters is (in_channels/g\*1\*1)\*out_channels.

```python
from mindspore import nn
import mindspore.ops as ops
from mindspore import Tensor

class GroupConv(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, pad_mode="pad", pad=0, groups=1, has_bias=False):
        super(GroupConv, self).__init__()
        self.groups = groups
        self.convs = nn.CellList()
        for _ in range(groups):
            self.convs.append(nn.Conv2d(in_channels // groups, out_channels // groups,
                                        kernel_size=kernel_size, stride=stride, has_bias=has_bias,
                                        padding=pad, pad_mode=pad_mode, group=1, weight_init='xavier_uniform'))

    def construct(self, x):
        features = ops.split(x, split_size_or_sections=int(len(x[0]) // self.groups), axis=1)
        outputs = ()
        for i in range(self.groups):
            outputs = outputs + (self.convs[i](features[i].astype("float32")),)
        out = ops.cat(outputs, axis=1)
        return out
```

### Channel Shuffle

The disadvantage of group convolution is that channels of different groups cannot communicate with each other. After the GConv layer is stacked, feature maps of different groups do not communicate with each other. The convolution seems to be divided into *g* irrelevant roads, **which may reduce the feature extraction capability of the network**. This is why networks such as Xception and MobileNet use dense pointwise convolution.

To solve the preceding problem, ShuffleNet optimizes a large number of dense 1x1 convolutions (the computing usage reaches 93.4%) and introduces the channel shuffle mechanism. This operation is to **evenly distribute and reassemble** different group channels so that the network can process the information of different groups of channels at the next layer.

![shufflenet3](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/application/source_zh_cn/cv/images/shufflenet_3.png)

As shown in the following figure, for *g* groups, each group has a feature map with *n* channels. First, reshape the feature map into a matrix with *g* rows and *n* columns, then transpose the matrix into *n* rows and *g* columns, and finally perform the flatten operation to obtain a new arrangement. These operations are differential and easy to calculate, which solves the problem of information interaction and complies with the lightweight feature of ShuffleNet lightweight network design.

![shufflenet4](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/application/source_zh_cn/cv/images/shufflenet_4.png)

For ease of reading, the code implementation of channel shuffle is placed in the code of the ShuffleNet modules below.

### ShuffleNet Modules

As shown in the following figure, ShuffleNet changes the Bottleneck structure in ResNet from (a) to (b) and (c).

1. Change the start and end $1\times 1$ convolution modules (dimension reduction and increase) to point wise group convolution.

2. To exchange information between different channels, perform channel shuffle after dimension reduction.

3. In the downsampling module, set the step of $3 \times 3$ depth wise convolution to 2, and reduce the length and width to half of the original values. Therefore, $3\times 3$ mean-pooling with the step of 2 is used in shortcut, and the addition is changed to concatenation.

![shufflenet5](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/application/source_zh_cn/cv/images/shufflenet_5.png)

```python
class ShuffleV1Block(nn.Cell):
    def __init__(self, inp, oup, group, first_group, mid_channels, ksize, stride):
        super(ShuffleV1Block, self).__init__()
        self.stride = stride
        pad = ksize // 2
        self.group = group
        if stride == 2:
            outputs = oup - inp
        else:
            outputs = oup
        self.relu = nn.ReLU()
        branch_main_1 = [
            GroupConv(in_channels=inp, out_channels=mid_channels,
                      kernel_size=1, stride=1, pad_mode="pad", pad=0,
                      groups=1 if first_group else group),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        ]
        branch_main_2 = [
            nn.Conv2d(mid_channels, mid_channels, kernel_size=ksize, stride=stride,
                      pad_mode='pad', padding=pad, group=mid_channels,
                      weight_init='xavier_uniform', has_bias=False),
            nn.BatchNorm2d(mid_channels),
            GroupConv(in_channels=mid_channels, out_channels=outputs,
                      kernel_size=1, stride=1, pad_mode="pad", pad=0,
                      groups=group),
            nn.BatchNorm2d(outputs),
        ]
        self.branch_main_1 = nn.SequentialCell(branch_main_1)
        self.branch_main_2 = nn.SequentialCell(branch_main_2)
        if stride == 2:
            self.branch_proj = nn.AvgPool2d(kernel_size=3, stride=2, pad_mode='same')

    def construct(self, old_x):
        left = old_x
        right = old_x
        out = old_x
        right = self.branch_main_1(right)
        if self.group > 1:
            right = self.channel_shuffle(right)
        right = self.branch_main_2(right)
        if self.stride == 1:
            out = self.relu(left + right)
        elif self.stride == 2:
            left = self.branch_proj(left)
            out = ops.cat((left, right), 1)
            out = self.relu(out)
        return out

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = ops.shape(x)
        group_channels = num_channels // self.group
        x = ops.reshape(x, (batchsize, group_channels, self.group, height, width))
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        x = ops.reshape(x, (batchsize, num_channels, height, width))
        return x
```

### Building a ShuffleNet

The following figure shows the ShuffleNet structure. The following uses the input image $224 \times 224$ and three groups (g = 3) as an example. First, pass through 24 convolutional layers whose convolution kernel size is $3 \times 3$ and stride is 2. The size of the output feature map is $112 \times 112$, and the channel is 24. Then, pass through the maximum pooling layer whose stride is 2. The size of the output feature map is $56 \times 56$, and the number of channels remains unchanged. Stack three ShuffleNet modules (stage 2, stage 3, and stage 4). The three modules are repeated four times, eight times, and four times respectively. Each module starts to pass through the downsampling module (that is, (c) in the preceding figure) to halve the length and width of the feature map and double the number of channels (except the downsampling module in stage 2, which changes the number of channels from 24 to 240). After the global mean-pooling is passed through, the output size is $1 \times 1 \times 960$. Then, the fully-connected layer and softmax are passed through to obtain the classification probability.

![shufflenet6](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/application/source_zh_cn/cv/images/shufflenet_6.png)

```python
class ShuffleNetV1(nn.Cell):
    def __init__(self, n_class=1000, model_size='2.0x', group=3):
        super(ShuffleNetV1, self).__init__()
        print('model size is ', model_size)
        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if group == 3:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 12, 120, 240, 480]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 240, 480, 960]
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 360, 720, 1440]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 480, 960, 1920]
            else:
                raise NotImplementedError
        elif group == 8:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 16, 192, 384, 768]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 384, 768, 1536]
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 576, 1152, 2304]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 768, 1536, 3072]
            else:
                raise NotImplementedError
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.SequentialCell(
            nn.Conv2d(3, input_channel, 3, 2, 'pad', 1, weight_init='xavier_uniform', has_bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                first_group = idxstage == 0 and i == 0
                features.append(ShuffleV1Block(input_channel, output_channel,
                                               group=group, first_group=first_group,
                                               mid_channels=output_channel // 4, ksize=3, stride=stride))
                input_channel = output_channel
        self.features = nn.SequentialCell(features)
        self.globalpool = nn.AvgPool2d(7)
        self.classifier = nn.Dense(self.stage_out_channels[-1], n_class)

    def construct(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.globalpool(x)
        x = ops.reshape(x, (-1, self.stage_out_channels[-1]))
        x = self.classifier(x)
        return x
```

## Model Training and Validation

The CIFAR-10 dataset is used to pre-train ShuffleNet.

### Preparing and Loading the Training Set

The CIFAR-10 dataset is used to pre-train ShuffleNet. CIFAR-10 has 60,000 32 x 32 color images, which are evenly divided into 10 classes. 50,000 images are used as the training set, and 10,000 images are used as the test set. The following example uses the `mindspore.dataset.Cifar10Dataset` API to download and load the CIFAR-10 training set. Currently, only the CIFAR-10 binary version is supported.

```python
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz"

download(url, "./dataset", kind="tar.gz", replace=True)
```

```python
import mindspore as ms
from mindspore.dataset import Cifar10Dataset
from mindspore.dataset import vision, transforms

def get_dataset(train_dataset_path, batch_size, usage):
    image_trans = []
    if usage == "train":
        image_trans = [
            vision.RandomCrop((32, 32), (4, 4, 4, 4)),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.Resize((224, 224)),
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            vision.HWC2CHW()
        ]
    elif usage == "test":
        image_trans = [
            vision.Resize((224, 224)),
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            vision.HWC2CHW()
        ]
    label_trans = transforms.TypeCast(ms.int32)
    dataset = Cifar10Dataset(train_dataset_path, usage=usage, shuffle=True)
    dataset = dataset.map(image_trans, 'image')
    dataset = dataset.map(label_trans, 'label')
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

dataset = get_dataset("./dataset/cifar-10-batches-bin", 128, "train")
batches_per_epoch = dataset.get_dataset_size()
```

### Model Training

This section uses randomly initialized parameters for pre-training. Call `ShuffleNetV1` to define the network, set the number of parameters to `"2.0x"`, and define the loss function as cross-entropy loss. After four `warmup` epochs, cosine annealing is used as the learning rate, and `Momentum` is used as the optimizer. Finally, the model, loss function, and optimizer are encapsulated in `model` by using the `Model` interface in `train.model`, and the network is trained by using `model.train()`. After `ModelCheckpoint`, `CheckpointConfig`, `TimeMonitor`, and `LossMonitor` are transferred to the callback function, the number of training epochs, loss, and time are printed, and the CKPT file is saved in the current directory.

```python
import time
import mindspore
import numpy as np
from mindspore import Tensor, nn
from mindspore.train import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor, Model, Top1CategoricalAccuracy, Top5CategoricalAccuracy

def train():
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")
    net = ShuffleNetV1(model_size="2.0x", n_class=10)
    loss = nn.CrossEntropyLoss(weight=None, reduction='mean', label_smoothing=0.1)
    min_lr = 0.0005
    base_lr = 0.05
    lr_scheduler = mindspore.nn.cosine_decay_lr(min_lr,
                                                base_lr,
                                                batches_per_epoch*250,
                                                batches_per_epoch,
                                                decay_epoch=250)
    lr = Tensor(lr_scheduler[-1])
    optimizer = nn.Momentum(params=net.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=0.00004, loss_scale=1024)
    loss_scale_manager = ms.amp.FixedLossScaleManager(1024, drop_overflow_update=False)
    model = Model(net, loss_fn=loss, optimizer=optimizer, amp_level="O3", loss_scale_manager=loss_scale_manager)
    callback = [TimeMonitor(), LossMonitor()]
    save_ckpt_path = "./"
    config_ckpt = CheckpointConfig(save_checkpoint_steps=batches_per_epoch, keep_checkpoint_max=5)
    ckpt_callback = ModelCheckpoint("shufflenetv1", directory=save_ckpt_path, config=config_ckpt)
    callback += [ckpt_callback]

    print("============== Starting Training ==============")
    start_time = time.time()
    model.train(250, dataset, callbacks=callback)
    use_time = time.time() - start_time
    hour = str(int(use_time // 60 // 60))
    minute = str(int(use_time // 60 % 60))
    second = str(int(use_time % 60))
    print("total time:" + hour + "h " + minute + "m " + second + "s")
    print("============== Train Success ==============")

if __name__ == '__main__':
    train()
```

Output:

```text
model size is  2.0x
============== Starting Training ==============
epoch: 1 step: 391, loss is 1.8377745151519775
epoch: 2 step: 391, loss is 1.825901403427124
epoch: 3 step: 391, loss is 1.8933873176574707
...                                        ...
epoch: 248 step: 391, loss is 0.6060634851455688
epoch: 249 step: 391, loss is 0.604820728302002
epoch: 250 step: 391, loss is 0.6010043621063232
Train epoch time: 305032.881 ms, per step time: 780.135 ms
total time:21h 4m 27s
============== Train Success ==============
```

The trained model is saved in `shufflenetv1-250_391.ckpt` of the current directory for evaluation.

### Model Evaluation

Evaluate the model on the CIFAR-10 test set.

After setting the path of the evaluation model, load the dataset, set the top 1 and top 5 evaluation metrics, and use the `model.eval()` interface to evaluate the model.

```python
from mindspore import load_checkpoint, load_param_into_net

def test():
    mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target="GPU")
    dataset = get_dataset("./dataset/cifar-10-batches-bin", 128, "test")
    net = ShuffleNetV1(model_size="2.0x", n_class=10)
    param_dict = load_checkpoint("shufflenetv1-250_391.ckpt")
    load_param_into_net(net, param_dict)
    net.set_train(False)
    loss = nn.CrossEntropyLoss(weight=None, reduction='mean', label_smoothing=0.1)
    eval_metrics = {'Loss': nn.Loss(), 'Top_1_Acc': Top1CategoricalAccuracy(),
                    'Top_5_Acc': Top5CategoricalAccuracy()}
    model = Model(net, loss_fn=loss, metrics=eval_metrics)
    start_time = time.time()
    res = model.eval(dataset, dataset_sink_mode=False)
    use_time = time.time() - start_time
    hour = str(int(use_time // 60 // 60))
    minute = str(int(use_time // 60 % 60))
    second = str(int(use_time % 60))
    log = "result:" + str(res) + ", ckpt:'" + "./shufflenetv1-250_391.ckpt" \
        + "', time: " + hour + "h " + minute + "m " + second + "s"
    print(log)
    filename = './eval_log.txt'
    with open(filename, 'a') as file_object:
        file_object.write(log + '\n')

if __name__ == '__main__':
    test()
```

Output:

```text
model size is  2.0x
result:{'Loss': 1.0217913215673422, 'Top_1_Acc': 0.8152, 'Top_5_Acc': 0.975}, ckpt:'./shufflenetv1-250_391.ckpt', time: 0h 0m 21s
```

### Model Prediction

Predict the model on the CIFAR-10 test set and visualize the prediction result.

```python
import mindspore
import matplotlib.pyplot as plt
import mindspore.dataset as ds

net = ShuffleNetV1(model_size="2.0x", n_class=10)
show_lst = []
param_dict = load_checkpoint("shufflenetv1-250_391.ckpt")
load_param_into_net(net, param_dict)
model = Model(net)
dataset_predict = ds.Cifar10Dataset(dataset_dir="./dataset/cifar-10-batches-bin", shuffle=False, usage="train")
dataset_show = ds.Cifar10Dataset(dataset_dir="./dataset/cifar-10-batches-bin", shuffle=False, usage="train")
dataset_show = dataset_show.batch(16)
show_images_lst = next(dataset_show.create_dict_iterator())["image"].asnumpy()
image_trans = [
    vision.RandomCrop((32, 32), (4, 4, 4, 4)),
    vision.RandomHorizontalFlip(prob=0.5),
    vision.Resize((224, 224)),
    vision.Rescale(1.0 / 255.0, 0.0),
    vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    vision.HWC2CHW()
        ]
dataset_predict = dataset_predict.map(image_trans, 'image')
dataset_predict = dataset_predict.batch(16)
class_dict = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
# Inference effect display (The upper part is the prediction result, and the lower part is the inference effect image.)
plt.figure(figsize=(16, 5))
predict_data = next(dataset_predict.create_dict_iterator())
output = model.predict(ms.Tensor(predict_data['image']))
pred = np.argmax(output.asnumpy(), axis=1)
index = 0
for image in show_images_lst:
    plt.subplot(2, 8, index+1)
    plt.title('{}'.format(class_dict[pred[index]]))
    index += 1
    plt.imshow(image)
    plt.axis("off")
plt.show()
```

Output:

```text
model size is  2.0x
```
