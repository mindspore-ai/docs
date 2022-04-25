# ResNet-50 Second-Order Optimization Practice

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/design/cv_resnet50_second_order_optimizer.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

## Overview

Common optimization algorithms are classified into the first-order and the second-order optimization algorithms. Typical first-order optimization algorithms, such as stochastic gradient descent (SGD), support a small amount of computation with high computation speed but a low convergence speed and require a large number of training steps. The second-order optimization algorithms use the second-order derivative of the objective function to accelerate convergence to the optimal value of a model, and require a small quantity of training steps. However, the second-order optimization algorithms have excessively high computation costs, an overall execution time of the second-order optimization algorithms is still slower than that of the first-order optimization algorithms. As a result, the second-order optimization algorithms are not widely used in deep neural network training. The main computation costs of the second-order optimization algorithms lie in the inverse operation of the second-order information matrices such as the Hessian matrix and the [Fisher information matrix (FIM)](https://arxiv.org/pdf/1808.07172.pdf). The time complexity is about $O(n^3)$.

Based on the existing natural gradient algorithm, MindSpore development team uses optimized acceleration methods such as approximation and sharding for the FIM, greatly reducing the computation complexity of the inverse matrix and developing the available second-order optimizer THOR. With eight Ascend 910 AI processors, THOR can complete the training of ResNet-50 v1.5 network and ImageNet dataset within 72 minutes, which is nearly twice the speed of SGD+Momentum.

This tutorial describes how to use the second-order optimizer THOR provided by MindSpore to train the ResNet-50 v1.5 network and ImageNet dataset on Ascend 910 and GPU.
> Download address of the complete code example:
<https://gitee.com/mindspore/models/tree/r1.7/official/cv/resnet>

Directory Structure of Code Examples

```text
├── resnet
    ├── README.md
    ├── scripts
        ├── run_distribute_train.sh         # launch distributed training for Ascend 910
        ├── run_eval.sh                     # launch inference for Ascend 910
        ├── run_distribute_train_gpu.sh     # launch distributed training for GPU
        ├── run_eval_gpu.sh                 # launch inference for GPU
    ├── src
        ├── dataset.py                      # data preprocessing
        ├── CrossEntropySmooth.py           # CrossEntropy loss function
        ├── lr_generator.py                 # generate learning rate for every step
        ├── resnet.py                       # ResNet50 backbone
        ├── model_utils
            ├── config.py                   # parameter configuration
    ├── eval.py                             # infer script
    ├── train.py                            # train script
```

The overall execution process is as follows:

1. Prepare the ImageNet dataset and process the required dataset.
2. Define the ResNet-50 network.
3. Define the loss function and the optimizer THOR.
4. Load the dataset and perform training. After the training is complete, check the result and save the model file.
5. Load the saved model for inference.

## Preparation

Ensure that MindSpore has been correctly installed. If not, install it by referring to [Install](https://www.mindspore.cn/install/en).

### Preparing the Dataset

Download the complete ImageNet2012 dataset, decompress the dataset, and save it to the `ImageNet2012/ilsvrc` and `ImageNet2012/ilsvrc_eval` directories in the local workspace.

The directory structure is as follows:

```text
└─ImageNet2012
    ├─ilsvrc
    │      n03676483
    │      n04067472
    │      n01622779
    │      ......
    └─ilsvrc_eval
    │      n03018349
    │      n02504013
    │      n07871810
    │      ......
```

### Configuring Distributed Environment Variables

#### Ascend 910

For details about how to configure the distributed environment variables of Ascend 910 AI processors, see [Parallel Distributed Training (Ascend)](https://www.mindspore.cn/tutorials/experts/en/r1.7/parallel/train_ascend.html#configuring-distributed-environment-variables).

#### GPU

For details about how to configure the distributed environment of GPUs, see [Parallel Distributed Training (GPU)](https://www.mindspore.cn/tutorials/experts/en/r1.7/parallel/train_gpu.html#configuring-distributed-environment).

## Loading the Dataset

During distributed training, load the dataset in parallel mode and process it through the data argumentation API provided by MindSpore. The `src/dataset.py` script in the source code is for loading and processing the dataset.

```python
import os
from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.communication import init, get_rank, get_group_size


def create_dataset2(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend", distribute=False,
                    enable_cache=False, cache_session_id=None):
    """
    Create a training or evaluation ImageNet2012 dataset for ResNet50.

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether the dataset is used for training or evaluation.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False
        enable_cache(bool): whether tensor caching service is used for evaluation. Default: False
        cache_session_id(int): if enable_cache is set, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        if distribute:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1

    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                         num_shards=device_num, shard_id=rank_id)

    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    # only enable cache for eval
    if do_train:
        enable_cache = False
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
        data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8,
                                cache=eval_cache)
    else:
        data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set
```

> MindSpore supports multiple data processing and augmentation operations. These operations are usually used in combination. For details, see [Data Processing](https://www.mindspore.cn/tutorials/en/r1.7/advanced/dataset.html).

## Defining the Network

Use the ResNet-50 v1.5 network model as an example. Define the [ResNet-50 network](https://gitee.com/mindspore/models/blob/r1.7/official/cv/resnet/src/resnet.py).

After the network is built, call the defined ResNet-50 in the `__main__` function.

```python
...
from src.resnet import resnet50 as resnet
...
if __name__ == "__main__":
    ...
    # define net
    net = resnet(class_num=config.class_num)
    ...
```

## Defining the Loss Function and Optimizer THOR

### Defining the Loss Function

Loss functions supported by MindSpore include `SoftmaxCrossEntropyWithLogits`, `L1Loss`, and `MSELoss`. The `SoftmaxCrossEntropyWithLogits` loss function is required by THOR.

The implementation procedure of the loss function is in the `src/CrossEntropySmooth.py` script. A common trick in deep network model training, label smoothing, is used to improve the model tolerance to error label classification by smoothing real labels, thereby improving the model generalization capability.

```python
class CrossEntropySmooth(LossBase):
    """CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = ops.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss
```

Call the defined loss function in the `__main__` function.

```python
...
from src.CrossEntropySmooth import CrossEntropySmooth
...
if __name__ == "__main__":
    ...
    # define the loss function
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(sparse=True, reduction="mean",
                              smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    ...
```

### Defining the Optimizer

The parameter update formula of THOR is as follows:

$$ \theta^{t+1} = \theta^t + \alpha F^{-1}\nabla E$$

The meanings of parameters in the formula are as follows:

- $\theta$: trainable parameters on the network.
- $t$: number of training steps.
- $\alpha$: learning rate, which is the parameter update value per step.
- $F^{-1}$: FIM obtained from the network computation.
- $\nabla E$: the first-order gradient value.

As shown in the parameter update formula, THOR needs to additionally compute an FIM of each layer. The FIM can adaptively adjust the parameter update step and direction of each layer, accelerating convergence and reducing parameter optimization complexity.

For more introduction of THOR optimizer, please see [THOR paper](https://www.aaai.org/AAAI21Papers/AAAI-6611.ChenM.pdf).

When calling the second-order optimizer THOR provided by MindSpore, THOR will automatically call the conversion interface to convert the Conv2d and Dense layers in the original network model into corresponding [Conv2dThor](https://gitee.com/mindspore/mindspore/blob/r1.7/mindspore/python/mindspore/nn/layer/thor_layer.py) and [DenseThor](https://gitee.com/mindspore/mindspore/blob/r1.7/mindspore/python/mindspore/nn/layer/thor_layer.py).
And the FIM of each layer is computed and saved in Conv2dThor and DenseThor.

> Compared to the original network model, conversion network model has the same backbone and weights.

```python
...
from mindspore.nn import thor
...
if __name__ == "__main__":
    ...
    # learning rate setting and damping setting
    from src.lr_generator import get_thor_lr, get_thor_damping
    lr = get_thor_lr(0, config.lr_init, config.lr_decay, config.lr_end_epoch, step_size, decay_epochs=39)
    damping = get_thor_damping(0, config.damping_init, config.damping_decay, 70, step_size)
    # define the optimizer
    split_indices = [26, 53]
    opt = thor(net, Tensor(lr), Tensor(damping), config.momentum, config.weight_decay, config.loss_scale,
               config.batch_size, split_indices=split_indices, frequency=config.frequency)
    ...
```

## Training the Network

### Saving the Configured Model

MindSpore provides the callback mechanism to execute customized logic during training. The `ModelCheckpoint` function provided by the framework is used in this example.
`ModelCheckpoint` can save the network model and parameters for subsequent fine-tuning.
`TimeMonitor` and `LossMonitor` are callback functions provided by MindSpore. They can be used to monitor the single training step time and `loss` value changes during training, respectively.

```python
...
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
...
if __name__ == "__main__":
    ...
    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]
    ...
```

### Configuring the Network Training

Use the `model.train` API provided by MindSpore to easily train the network. THOR reduces the computation workload and improves the computation speed by reducing the frequency of updating the second-order matrix. Therefore, the [ModelThor](https://gitee.com/mindspore/mindspore/blob/r1.7/mindspore/python/mindspore/train/train_thor/model_thor.py) class is redefined to inherit the Model class provided by MindSpore. The parameter of THOR for controlling the frequency of updating the second-order matrix can be obtained by the ModelThor class. You can adjust this parameter to optimize the overall performance.
MindSpore provides a one-click conversion interface from Model class to ModelThor class.

```python
...
from mindspore import FixedLossScaleManager
from mindspore import Model
from mindspore.train.train_thor import ConvertModelUtils
...

if __name__ == "__main__":
    ...
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                  amp_level="O2", keep_batchnorm_fp32=False, eval_network=dist_eval_network)
    if cfg.optimizer == "Thor":
        model = ConvertModelUtils().convert_to_thor_model(model=model, network=net, loss_fn=loss, optimizer=opt,
                                                          loss_scale_manager=loss_scale, metrics={'acc'},
                                                          amp_level="O2", keep_batchnorm_fp32=False)  
    ...
```

### Running the Script

After the training script is defined, call the shell script in the `scripts` directory to start the distributed training process.

#### Ascend 910

Currently, MindSpore distributed execution on Ascend uses the single-device single-process running mode. That is, one process runs on one device, and the number of total processes is the same as the number of devices that are being used. All processes are executed in the background. Create a directory named `train_parallel`+`device_id` for each process to store log information, operator compilation information, and training checkpoint files. The following takes the distributed training script for eight devices as an example to describe how to run the script.

Run the script:

```bash
bash run_distribute_train.sh <RANK_TABLE_FILE> <DATASET_PATH> <CONFIG_PATH>
```

Variables `RANK_TABLE_FILE`, `DATASET_PATH` and `CONFIG_PATH` need to be transferred to the script. The meanings of variables are as follows:

- `RANK_TABLE_FILE`: path for storing the networking information file (about the rank table file, you can refer to [HCCL_TOOL](https://gitee.com/mindspore/models/tree/r1.7/utils/hccl_tools))
- `DATASET_PATH`: training dataset path
- `CONFIG_PATH`: config file path

For details about other environment variables, see configuration items in the installation guide.

The following is an example of loss values output during training:

```text
...
epoch: 1 step: 5004, loss is 4.4182425
epoch: 2 step: 5004, loss is 3.740064
epoch: 3 step: 5004, loss is 4.0546017
epoch: 4 step: 5004, loss is 3.7598825
epoch: 5 step: 5004, loss is 3.3744206
...
epoch: 40 step: 5004, loss is 1.6907625
epoch: 41 step: 5004, loss is 1.8217756
epoch: 42 step: 5004, loss is 1.6453942
...
```

After the training is complete, the checkpoint file generated by each device is stored in the training directory. The following is an example of the checkpoint file generated by `device_0`:

```text
└─train_parallel0
    ├─ckpt_0
        ├─resnet-1_5004.ckpt
        ├─resnet-2_5004.ckpt
        │      ......
        ├─resnet-42_5004.ckpt
        │      ......
```

In the preceding information,
`*.ckpt` indicates the saved model parameter file. The name of a checkpoint file is in the following format: *Network name*-*Number of epochs*_*Number of steps*.ckpt.

#### GPU

On the GPU hardware platform, MindSpore uses `mpirun` of OpenMPI to perform distributed training. The process creates a directory named `train_parallel` to store log information and training checkpoint files. The following takes the distributed training script for eight devices as an example to describe how to run the script.
Run the script:

```bash
bash run_distribute_train_gpu.sh <DATASET_PATH> <CONFIG_PATH>
```

Variables `DATASET_PATH` and `CONFIG_PATH` need to be transferred to the script. The meanings of variables are as follows:

- `DATASET_PATH`: training dataset path
- `CONFIG_PATH`: config file path

During GPU-based training, the `DEVICE_ID` environment variable is not required. Therefore, you do not need to call `int(os.getenv('DEVICE_ID'))` in the main training script to obtain the device ID or transfer `device_id` to `context`. You need to set `device_target` to `GPU` and call `init()` to enable the NCCL.

The following is an example of loss values output during training:

```text
...
epoch: 1 step: 5004, loss is 4.2546034
epoch: 2 step: 5004, loss is 4.0819564
epoch: 3 step: 5004, loss is 3.7005644
epoch: 4 step: 5004, loss is 3.2668946
epoch: 5 step: 5004, loss is 3.023509
...
epoch: 36 step: 5004, loss is 1.645802
...
```

The following is an example of model files saved after training:

```text
└─train_parallel
    ├─ckpt_0
        ├─resnet-1_5004.ckpt
        ├─resnet-2_5004.ckpt
        │      ......
        ├─resnet-36_5004.ckpt
        │      ......
    ......
    ├─ckpt_7
        ├─resnet-1_5004.ckpt
        ├─resnet-2_5004.ckpt
        │      ......
        ├─resnet-36_5004.ckpt
        │      ......
```

## Model Inference

Use the checkpoint files saved during training to perform inference and validate the model generalization capability. Load the model file using the `load_checkpoint` API, call the `eval` API of the `Model` to predict the input image class, and compare the predicted class with the actual class of the input image to obtain the final prediction accuracy.

### Defining the Inference Network

1. Use the `load_checkpoint` API to load the model file.
2. Use the `model.eval` API to read the test dataset for inference.
3. Compute the prediction accuracy.

```python
...
from mindspore import load_checkpoint, load_param_into_net
...

if __name__ == "__main__":
    ...
    # define net
    net = resnet(class_num=config.class_num)

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss
    if args_opt.dataset == "imagenet2012":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction='mean',
                                  smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
    ...
```

### Inference

After the inference network is defined, the shell script in the `scripts` directory is called for inference.

#### Ascend 910

On the Ascend 910 hardware platform, run the following inference command:

```bash
bash run_eval.sh <DATASET_PATH> <CHECKPOINT_PATH>
```

Variables `DATASET_PATH`, `CHECKPOINT_PATH` and `CONFIG_PATH` need to be transferred to the script. The meanings of variables are as follows:

- `DATASET_PATH`: inference dataset path
- `CHECKPOINT_PATH`: path for storing the checkpoint file
- `CONFIG_PATH`: config file path

Currently, a single device (device 0 by default) is used for inference. The inference result is as follows:

```text
result: {'top_5_accuracy': 0.9295574583866837, 'top_1_accuracy': 0.761443661971831} ckpt=train_parallel0/resnet-42_5004.ckpt
```

- `top_5_accuracy`: For an input image, if the labels whose prediction probability ranks top 5 match actual labels, the classification is correct.
- `top_1_accuracy`: For an input image, if the label with the highest prediction probability is the same as the actual label, the classification is correct.

#### GPU

On the GPU hardware platform, run the following inference command:

```bash
 bash run_eval_gpu.sh <DATASET_PATH> <CHECKPOINT_PATH>
```

Variables `DATASET_PATH`, `CHECKPOINT_PATH` and `CONFIG_PATH` need to be transferred to the script. The meanings of variables are as follows:

- `DATASET_PATH`: inference dataset path
- `CHECKPOINT_PATH`: path for storing the checkpoint file
- `CONFIG_PATH`: config file path

The inference result is as follows:

```text
result: {'top_5_accuracy': 0.9287972151088348, 'top_1_accuracy': 0.7597031049935979} ckpt=train_parallel/resnet-36_5004.ckpt
```
