[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/mixed_precision.md)

[Introduction](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/master/beginner/quick_start.html#) || [Tensor](https://www.mindspore.cn/tutorials/en/master/beginner/tensor.html) || [Data Loading and Processing](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html) || [Model](https://www.mindspore.cn/tutorials/en/master/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/master/beginner/autograd.html) || [Train](https://www.mindspore.cn/tutorials/en/master/beginner/train.html) || [Save and Load](https://www.mindspore.cn/tutorials/en/master/beginner/save_load.html) || [Accelerating with Static Graphs](https://www.mindspore.cn/tutorials/en/master/beginner/accelerate_with_static_graph.html)|| **Mixed Precision**

# Automatic Mix Precision

Mixed precision training is a computing strategy that uses different numerical precision for different operations of the neural network during training. In neural network operations, some operations are not sensitive to numerical precision, and using lower precision can achieve significant acceleration (such as conv, matmul), while some of the operations usually need to retain high precision to ensure the correctness of the results due to the large difference between the input and output values (such as log, softmax).

The hardware acceleration modules are usually designed on current AI accelerator cards for targeting computationally intensive, precision-insensitive operations, such as TensorCore for NVIDIA GPUs and Cube for Ascend NPU. For neural networks with a larger share of operations, such as conv, matmul, their training speed usually has a larger acceleration ratio.

The `mindspore.amp` module provides a convenient interface for automatic mixed precision, allowing users to obtain training acceleration at different hardware backends with simple interface calls. In the following, we introduce the calculation principle of mixed precision, and then introduce the automatic mixed precision usage of MindSpore by example.

## Principle of Mixed Precision Calculation

Floating-point data types include double-precision (FP64), single-precision (FP32), and half-precision (FP16). In a training process of a neural network model, an FP32 data type is generally used by default to indicate a network model weight and other parameters. The following is a brief introduction to floating-point data types.

According to [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754), floating-point data types are classified into double-precision (FP64), single-precision (FP32), and half-precision (FP16). Each type is represented by three different bits. FP64 indicates a data type that uses 8 bytes (64 bits in total) for encoding and storage. FP32 indicates a data type that uses 4 bytes (32 bits in total) and FP16 indicates a data type that uses 2 bytes (16 bits in total). As shown in the following figure:

![fp16_vs_FP32](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_en/beginner/images/fp16_vs_fp32.png)

As shown in the figure, the storage space of FP16 is half that of FP32. Similarly, the storage space of FP32 is half that of FP64. Therefore, using FP16 for computing has the following advantages:

- Reduce memory usage: The bit width of FP16 is half that of FP32, so the memory used for parameters such as weights is also half of the original, saving memory for larger network models or training with more data.
- Higher computational efficiency: On special AI-accelerated chips such as Huawei Atlas training series and Atlas 200/300/500 inference product series, or GPUs on NVIDIA VOLTA architecture, execution performance is faster using FP16 than FP32.
- Accelerate communication efficiency: For distributed training, especially in the process of training large models, the communication overhead constrains the overall performance of network model training. Less bit-width of communication means that communication performance can be improved, waiting time can be reduced, and the flow of data can be accelerated.

But the use of FP16 also poses a number of problems:

- Data overflow: The valid data representation range for FP16 is $[5.9\times10^{-8}, 65504]$ and for FP32 is $[1.4\times10^{-45}, 1.7\times10^{38}]$. It can be seen that the effective range of FP16 is much narrower than that of FP32, and using FP16 to replace FP32 will result in overflow and underflow. In deep learning, the gradient (first-order derivative) of the weights in the network model needs to be calculated, so the gradient will be even smaller than the weight value and often prone to underflow.
- Rounding error: Rounding Error is when the backward gradient of the network model is small, which is generally represented by FP32. But the conversion to FP16 will be smaller than the minimum interval in the current interval and will lead to data overflow. If `0.00006666666` can be expressed normally in FP32, it will be expressed as `0.000067` after conversion to FP16, and the numbers that do not meet the minimum interval of FP16 will be forced to be rounded.

Therefore, the solution of the FP16 introduction problem needs to be considered while using mixed precision to obtain training speedup and memory savings. Loss Scale, a solution to the FP16 type data overflow problem, expands the loss by a certain number of times when calculating the loss value loss. According to the chain rule, the gradient is expanded accordingly and then scaled down by a corresponding multiple when the optimizer updates the weights, thus avoiding data underflow.

Based on the principles described above, a typical mixed precision computation process is shown in the following figure:

![mix precision](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/beginner/images/mix_precision_fp16.png)

1. Parameters stored in FP32.
2. During forward computation, when it comes to FP16 operators, the operator inputs and parameters need to be cast from FP32 to FP16 for computation.
3. Set the Loss layer to FP32 for computation.
4. During the inverse computation, the Loss Scale value is first multiplied to avoid underflow due to a too small inverse gradient.
5. FP16 parameters are involved in the gradient computation and their results will be cast back to FP32.
6. Dividing by the Loss scale value to restore the amplified gradient.
7. Determine if there is an overflow in the gradient, and skip the update if there is an overflow, otherwise the optimizer updates the original parameters with FP32.

In the following, we demonstrate the automatic mixed precision implementation of MindSpore by importing the handwritten digit recognition model and dataset from [Quick Start](https://www.mindspore.cn/tutorials/en/master/beginner/quick_start.html).

```python
import mindspore as ms
from mindspore import nn
from mindspore import value_and_grad
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset

# Download data from open datasets
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)


def datapipe(path, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(ms.int32)

    dataset = MnistDataset(path)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = datapipe('MNIST_Data/train', 64)

# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits
```

```text
Downloading data from https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip (10.3 MB)

file_sizes: 100%|██████████████████████████| 10.8M/10.8M [00:07<00:00, 1.53MB/s]
Extracting zip file...
Successfully downloaded / unzipped to ./
```

## Type Conversions

Mixed precision calculations require type conversion of operations that require low precision, converting their input to FP16 types, and then converting them back to FP32 types after the output is obtained. MindSpore provides both automatic and manual type conversion methods to meet the different needs for ease of use and flexibility, which are described below.

### Automatic Type Conversion

The `mindspore.amp.auto_mixed_precision` interface provides the function to do automatic type conversion for networks. Automatic type conversion follows a blacklist and white list mechanism with five levels configured according to common operator precision conventions, as follows:

- 'O0': Neural network keeps FP32.
- 'O1': Operation cast to FP16 by whitelist.
- 'O2': Retain FP32 by blacklist and the rest of operations cast to FP16.
- 'O3': The neural network is fully cast to FP16.
- 'auto': Cast operators in whitelist to FP16, cast operators in blacklist to FP32, and the rest of operators choose the highest floating-point precision of operator input for conversion.

The following is an example of using automatic type conversion:

```python
from mindspore.amp import auto_mixed_precision

model = Network()
model = auto_mixed_precision(model, 'O2')
```

### Manual Type Conversion

Usually automatic type conversion can be used to satisfy most of the mixed precision training needs. But when users need to finely control the precision of operations in different parts of the neural network, they can be controlled by means of manual type conversion.

> Manual type conversions need to take into account the precision of each module in the model and are generally used only when extreme performance is required.

Below we adapt `Network` in the previous article to demonstrate different ways of manual type conversion.

#### Cell Granularity Type Conversion

The `nn.Cell` class provides the `to_float` method to configure the module's operator precision with a single click, automatically casting the module input to the specified precision.

```python
class NetworkFP16(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512).to_float(ms.float16),
            nn.ReLU(),
            nn.Dense(512, 512).to_float(ms.float16),
            nn.ReLU(),
            nn.Dense(512, 10).to_float(ms.float16)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits
```

#### Custom Granularity Type Conversion

When the user needs to configure the precision of operations in a single operation, or a combination of multiple modules, Cell granularity often can not meet the purpose of custom granularity control by directly casting the type of input data.

```python
class NetworkFP16Manual(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        x = x.astype(ms.float16)
        logits = self.dense_relu_sequential(x)
        logits = logits.astype(ms.float32)
        return logits
```

## Loss Scaling

Two implementations of Loss Scale are provided in MindSpore, [mindspore.amp.StaticLossScaler](https://www.mindspore.cn/docs/en/master/api_python/amp/mindspore.amp.StaticLossScaler.html) and [mindspore.amp.DynamicLossScalar](https://www.mindspore.cn/docs/en/master/api_python/amp/mindspore.amp.DynamicLossScalar.html), whose difference is whether the loss scale value is dynamically adjusted. The following is an example of `DynamicLossScalar`, which implements the neural network training logic according to the mixed precision calculation process.

First, instantiate the LossScaler and manually scale up the loss value when defining the forward network.

```python
from mindspore.amp import DynamicLossScaler

# Instantiate loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)

# Define LossScaler
loss_scaler = DynamicLossScaler(scale_value=2**16, scale_factor=2, scale_window=50)

def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    # scale up the loss value
    loss = loss_scaler.scale(loss)
    return loss, logits
```

Next, a function transformation is performed to obtain the gradient function.

```python
grad_fn = value_and_grad(forward_fn, None, model.trainable_params(), has_aux=True)
```

Define the training step: Calculates the current gradient value and recovers the loss. Use `all_finite` to determine if there is a gradient underflow problem. If there is no overflow, restore the gradient and update the network weight, while if there is overflow, skip this step.

```python
from mindspore.amp import all_finite

@ms.jit
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    loss = loss_scaler.unscale(loss)

    is_finite = all_finite(grads)
    if is_finite:
        grads = loss_scaler.unscale(grads)
        optimizer(grads)
    loss_scaler.adjust(is_finite)

    return loss
```

Finally, we train 1 epoch and observe the convergence of the loss trained using automatic mixed precision.

```python
size = train_dataset.get_dataset_size()
model.set_train()
for batch, (data, label) in enumerate(train_dataset.create_tuple_iterator()):
    loss = train_step(data, label)

    if batch % 100 == 0:
        loss, current = loss.asnumpy(), batch
        print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
```

```text
loss: 2.305425  [  0/938]
loss: 2.289585  [100/938]
loss: 2.259094  [200/938]
loss: 2.176874  [300/938]
loss: 1.856715  [400/938]
loss: 1.398342  [500/938]
loss: 0.889620  [600/938]
loss: 0.709884  [700/938]
loss: 0.750509  [800/938]
loss: 0.482525  [900/938]
```

It can be seen that the loss convergence is normal and there is no overflow problem.

## Automatic Mixed Precision for `Cell` Configuration

MindSpore supports a programming paradigm that uses Cell to encapsulate the full computational graph. When the [mindspore.amp.build_train_network](https://www.mindspore.cn/docs/en/master/api_python/amp/mindspore.amp.build_train_network.html) interface can be used to automatically perform the type conversion and pass in the Loss Scale as part of the full graph computation. At this point, you only need to configure the mixed precision level and `LossScaleManager` to get the computational graph with the configured automatic mixed precision.

[mindspore.amp.FixedLossScaleManager](https://www.mindspore.cn/docs/en/master/api_python/amp/mindspore.amp.FixedLossScaleManager.html) and [mindspore.amp.DynamicLossScaleManager](https://www.mindspore.cn/docs/en/master/api_python/amp/mindspore.amp.DynamicLossScaleManager.html) are the Loss scale management interfaces for configuring the automatic mixed precision with `Cell`, corresponding to `StaticLossScalar` and `DynamicLossScalar`, respectively. For detailed information, refer to [mindspore.amp](https://www.mindspore.cn/docs/en/master/api_python/mindspore.amp.html).

> Automated mixed precision training with `Cell` configuration supports only `GPU` and `Ascend`.

```python
from mindspore.amp import build_train_network, FixedLossScaleManager

model = Network()
loss_scale_manager = FixedLossScaleManager()

model = build_train_network(model, optimizer, loss_fn, level="O2", loss_scale_manager=loss_scale_manager)
```

## `Model` Configures Automatic Mixed Precision

[mindspore.train.Model](https://www.mindspore.cn/docs/en/master/api_python/train/mindspore.train.Model.html) is a high level encapsulation for fast training of neural networks, which encapsulates `mindspore.amp.build_train_network`, so again, only the mixed precision level and `LossScaleManager` need to be configured for automatic mixed precision training.

> Automated mixed precision training with `Model` configuration supports only `GPU` and `Ascend`.

```python
from mindspore.train import Model, LossMonitor
# Initialize network
model = Network()
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)

loss_scale_manager = FixedLossScaleManager()
trainer = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'}, amp_level="O2", loss_scale_manager=loss_scale_manager)

loss_callback = LossMonitor(100)
trainer.train(10, train_dataset, callbacks=[loss_callback])
```

```text
epoch: 1 step: 100, loss is 2.2883859
epoch: 1 step: 200, loss is 2.2612116
epoch: 1 step: 300, loss is 2.1563218
epoch: 1 step: 400, loss is 1.9420109
epoch: 1 step: 500, loss is 1.396821
epoch: 1 step: 600, loss is 1.0450488
epoch: 1 step: 700, loss is 0.69754004
epoch: 1 step: 800, loss is 0.6924556
epoch: 1 step: 900, loss is 0.57444984
...
epoch: 10 step: 58, loss is 0.13086069
epoch: 10 step: 158, loss is 0.07224723
epoch: 10 step: 258, loss is 0.08281057
epoch: 10 step: 358, loss is 0.09759849
epoch: 10 step: 458, loss is 0.17265382
epoch: 10 step: 558, loss is 0.10023793
epoch: 10 step: 658, loss is 0.08235697
epoch: 10 step: 758, loss is 0.10531154
epoch: 10 step: 858, loss is 0.19084263
```

> The image is quoted from [automatic-mixed-precision](https://developer.nvidia.com/automatic-mixed-precision).
