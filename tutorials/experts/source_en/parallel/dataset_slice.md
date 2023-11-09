# Dataset Slicing

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_en/parallel/dataset_slice.md)

## Overview

When performing distributed training, taking image data as an example, when the size of a single image is too large, such as large-format images of remote sensing satellites, even one image is too large, it is necessary to slice the images and read a portion of each card to perform distributed training. Scenarios that deal with dataset slicing need to be combined with model parallelism to achieve the desired effect of reducing video memory, so this feature is provided based on automatic parallelism. The sample used in this tutorial is not a large-format network, and is intended as an example only. Real-life applications to large-format networks often require detailed design of parallel strategies.

> Dataset sharding is only supported in fully-automatic mode and semi-automatic mode, and is not involved in data parallel mode.

Related interfaces:

1. `mindspore.dataset.vision.SlicePatches(num_height=1, num_width=1)`: Slices the Tensor into multiple blocks horizontally and vertically. Suitable for scenarios where the Tensor has a large height and width. `num_height` is the number of slices in vertical direction and `num_width` is the number of slices in horizontal direction. More parameters can be found in [SlicePatches](https://www.mindspore.cn/docs/en/r2.3/api_python/dataset_vision/mindspore.dataset.vision.SlicePatches.html).

2. `mindspore.set_auto_parallel_context(dataset_strategy=((1, 1, 1, 8), (8,))))`: indicates dataset slicing strategy. The `dataset_strategy` interface has the following limitations:

    - Each input is allowed to be sliced in at most one dimension. If `set_auto_parallel_context(dataset_strategy=((1, 1, 1, 8), (8,))))` or `dataset_strategy=((1, 1, 1, 8), (1,)))` is supported, each input is sliced in just one dimension, but not `dataset_strategy=((1, 1, 4, 2), (1,))`, whose first input is sliced into two dimensions.

    - The input with the highest dimension, the number of slices, must be more than the other dimensions. If `dataset_strategy=((1, 1, 1, 8), (8,)))` or `dataset_strategy=((1, 1, 1, 1, 1), (1,)))` is supported, the input with the most dimensions is the first one, with the number of slices of 8, and the rest of the inputs have no more than 8 slices, but not `dataset_ strategy=((1, 1, 1, 1), (8,))`, whose input with the most dimensions is the first input with a slice of 1, but the second input has a cut of 8, which exceeds the slices of the first input.

## Operation Practices

### Sample Code Description

> Download the full sample code here: [dataset_slice](https://gitee.com/mindspore/docs/tree/r2.3/docs/sample_code/dataset_slice).

The directory structure is as follows:

```text
└─ sample_code
    ├─ dataset_slice
       ├── train.py
       └── run.sh
    ...
```

`train.py` is the script that defines the network structure and the training process. `run.sh` is the execution script.

### Configuring a Distributed Environment

Specify the run mode, run device, run card number via the context interface. The parallel mode is semi-parallel mode and initializes HCCL or NCCL communication with init. In addition, configure the dataset `dataset_strategy` sharding strategy as ((1, 1, 1, 4), (1,)), which represents no slicing vertically and 4 slices horizontally.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
init()
slice_h_num = 1
slice_w_num = 4
ms.set_auto_parallel_context(dataset_strategy=((1, 1, slice_h_num, slice_w_num), (1,)))
```

### Loading the Dataset

When using dataset slicing, you need to call the `SlicePatches` interface to construct the dataset at the same time. To ensure that the read-in data is consistent across cards, the dataset needs to be fixed with a random number seed.

```python
import os
import mindspore.dataset as ds
from mindspore import nn

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

### Network Definition

The network definition here is consistent with the single-card model:

```python
from mindspore import nn

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

net = Network()
```

### Training the Network

In this step, the loss function, the optimizer, and the training process need to be defined, and this example is written in a functional way, which is partially consistent with the single-card model:

```python
from mindspore import nn, ops

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()

def forward_fn(data, target):
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

for epoch in range(1):
    i = 0
    for image, label in data_set:
        (loss_value, _), grads = grad_fn(image, label)
        optimizer(grads)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_value))
        i += 1
```

### Running Stand-alone 8-card Script

Next, the corresponding script is called by the command. Take the `mpirun` startup method, the 8-card distributed training script as an example, and perform the distributed training:

```bash
bash run.sh
```

After training, the log file is saved to the `log_output` directory, and the part of results about the Loss are saved in `log_output/1/rank.*/stdout`. The example is as follows:

```text
epoch: 0, step: 0, loss is 2.290361
epoch: 0, step: 10, loss is 1.9397954
epoch: 0, step: 20, loss is 1.4175975
epoch: 0, step: 30, loss is 1.0338318
epoch: 0, step: 40, loss is 0.64065826
epoch: 0, step: 50, loss is 0.8479157
...
```
