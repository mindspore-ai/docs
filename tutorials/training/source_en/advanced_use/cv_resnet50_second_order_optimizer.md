# ResNet-50 Second-Order Optimization Practice

`Linux` `Ascend` `GPU` `Model Development` `Model Optimization` `Expert`
<a href="https://gitee.com/mindspore/docs/blob/r1.1/tutorials/training/source_en/advanced_use/cv_resnet50_second_order_optimizer.md" target="_blank"><img src="../_static/logo_source.png"></a>&nbsp;&nbsp;

## Overview

Common optimization algorithms are classified into the first-order and the second-order optimization algorithms. Typical first-order optimization algorithms, such as stochastic gradient descent (SGD), support a small amount of computation with high computation speed but a low convergence speed and require a large number of training steps. The second-order optimization algorithms use the second-order derivative of the objective function to accelerate convergence to the optimal value of a model, and require a small quantity of training steps. However, the second-order optimization algorithms have excessively high computation costs, an overall execution time of the second-order optimization algorithms is still slower than that of the first-order optimization algorithms. As a result, the second-order optimization algorithms are not widely used in deep neural network training. The main computation costs of the second-order optimization algorithms lie in the inverse operation of the second-order information matrices such as the Hessian matrix and the [Fisher information matrix (FIM)](https://arxiv.org/pdf/1808.07172.pdf). The time complexity is about $O(n^3)$.

Based on the existing natural gradient algorithm, MindSpore development team uses optimized acceleration methods such as approximation and sharding for the FIM, greatly reducing the computation complexity of the inverse matrix and developing the available second-order optimizer THOR. With eight Ascend 910 AI processors, THOR can complete the training of ResNet-50 v1.5 network and ImageNet dataset within 72 minutes, which is nearly twice the speed of SGD+Momentum.

This tutorial describes how to use the second-order optimizer THOR provided by MindSpore to train the ResNet-50 v1.5 network and ImageNet dataset on Ascend 910 and GPU.
> Download address of the complete code example:
<https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/cv/resnet_thor>

Directory Structure of Code Examples

```shell
├── resnet_thor
    ├── README.md
    ├── scripts
        ├── run_distribute_train.sh         # launch distributed training for Ascend 910
        └── run_eval.sh                     # launch inference for Ascend 910
        ├── run_distribute_train_gpu.sh     # launch distributed training for GPU
        └── run_eval_gpu.sh                 # launch inference for GPU
    ├── src
        ├── crossentropy.py                 # CrossEntropy loss function
        ├── config.py                       # parameter configuration
        ├── dataset_helper.py               # dataset helper for minddata dataset
        ├── grad_reducer_thor.py            # grad reduce for thor
        ├── model_thor.py                   # model for train
        ├── resnet_thor.py                  # resnet50_thor backone
        ├── thor.py                         # thor optimizer
        ├── thor_layer.py                   # thor layer
        └── dataset.py                      # data preprocessing
    ├── eval.py                             # infer script
    ├── train.py                            # train script
    ├── export.py                           # export checkpoint file into air file
    └── mindspore_hub_conf.py               # config file for mindspore hub repository
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

For details about how to configure the distributed environment variables of Ascend 910 AI processors, see [Parallel Distributed Training (Ascend)](https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/distributed_training_ascend.html#configuring-distributed-environment-variables).

#### GPU

For details about how to configure the distributed environment of GPUs, see [Parallel Distributed Training (GPU)](https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/distributed_training_gpu.html#configuring-distributed-environment-variables).

## Loading the Dataset

During distributed training, load the dataset in parallel mode and process it through the data argumentation API provided by MindSpore. The `src/dataset.py` script in the source code is for loading and processing the dataset.

```python
import os
from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.communication.management import init, get_rank, get_group_size

def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend"):
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
        num_parallels = 8
    else:
        init()
        rank_id = get_rank()
        device_num = get_group_size()
        num_parallels = 4

    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallels, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallels, shuffle=True,
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

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=num_parallels)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallels)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set
```

> MindSpore supports multiple data processing and augmentation operations. These operations are usually used in combination. For details, see [Data Processing](https://www.mindspore.cn/tutorial/training/en/r1.1/use/data_preparation.html).

## Defining the Network

Use the ResNet-50 v1.5 network model as an example. Define the [ResNet-50 network](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/resnet/src/resnet.py), and replace the `Conv2d` and `Dense` operators with the operators customized by the second-order optimizer.
 The defined network model is stored in the `src/resnet_thor.py` script in the source code, and the customized operators `Conv2d_thor` and `Dense_thor` are stored in the `src/thor_layer.py` script.

- Use `Conv2d_thor` to replace `Conv2d` in the original network model.
- Use `Dense_thor` to replace `Dense` in the original network model.

> The `Conv2d_thor` and `Dense_thor` operators customized by THOR are used to save the second-order matrix information in model training. The backbone of the newly defined network is the same as that of the original network model.

After the network is built, call the defined ResNet-50 in the `__main__` function.

```python
...
from src.resnet_thor import resnet50
...
if __name__ == "__main__":
    ...
    # define the net
    net = resnet50(class_num=config.class_num, damping=damping, loss_scale=config.loss_scale,
                   frequency=config.frequency, batch_size=config.batch_size)
    ...
```

## Defining the Loss Function and Optimizer THOR

### Defining the Loss Function

Loss functions supported by MindSpore include `SoftmaxCrossEntropyWithLogits`, `L1Loss`, and `MSELoss`. The `SoftmaxCrossEntropyWithLogits` loss function is required by THOR.

The implementation procedure of the loss function is in the `src/crossentropy.py` script. A common trick in deep network model training, label smoothing, is used to improve the model tolerance to error label classification by smoothing real labels, thereby improving the model generalization capability.

```python
class CrossEntropy(_Loss):
    """CrossEntropy"""
    def __init__(self, smooth_factor=0., num_classes=1000):
        super(CrossEntropy, self).__init__()
        self.onehot = ops.OneHot()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.mean = ops.ReduceMean(False)

    def construct(self, logit, label):
        one_hot_label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, one_hot_label)
        loss = self.mean(loss, 0)
        return loss
```

Call the defined loss function in the `__main__` function.

```python
...
from src.crossentropy import CrossEntropy
...
if __name__ == "__main__":
    ...
    # define the loss function
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
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

As shown in the parameter update formula, THOR needs to additionally compute an FIM of each layer, and the FIM of each layer is obtained through computation in the customized network model. The FIM can adaptively adjust the parameter update step and direction of each layer, accelerating convergence and reducing parameter optimization complexity.

```python
...
if args_opt.device_target == "Ascend":
    from src.thor import THOR
else:
    from src.thor import THOR_GPU as THOR
...

if __name__ == "__main__":
    ...
    # learning rate setting
    lr = get_model_lr(0, config.lr_init, config.lr_decay, config.lr_end_epoch, step_size, decay_epochs=39)
    # define the optimizer
    opt = THOR(filter(lambda x: x.requires_grad, net.get_parameters()), Tensor(lr), config.momentum,
               filter(lambda x: 'matrix_A' in x.name, net.get_parameters()),
               filter(lambda x: 'matrix_G' in x.name, net.get_parameters()),
               filter(lambda x: 'A_inv_max' in x.name, net.get_parameters()),
               filter(lambda x: 'G_inv_max' in x.name, net.get_parameters()),
               config.weight_decay, config.loss_scale)
    ...
```

## Training the Network

### Saving the Configured Model

MindSpore provides the callback mechanism to execute customized logic during training. The `ModelCheckpoint` function provided by the framework is used in this example.
`ModelCheckpoint` can save the network model and parameters for subsequent fine-tuning.
`TimeMonitor` and `LossMonitor` are callback functions provided by MindSpore. They can be used to monitor the single training step time and `loss` value changes during training, respectively.

```python
...
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
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

Use the `model.train` API provided by MindSpore to easily train the network. THOR reduces the computation workload and improves the computation speed by reducing the frequency of updating the second-order matrix. Therefore, the Model_Thor class is redefined to inherit the Model class provided by MindSpore. The parameter for controlling the frequency of updating the second-order matrix is added to the Model_Thor class. You can adjust this parameter to optimize the overall performance.

```python
...
from mindspore import FixedLossScaleManager
from src.model_thor import Model_Thor as Model
...

if __name__ == "__main__":
    ...
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    if target == "Ascend":
        model = Model(net, loss_fn=loss, optimizer=opt, amp_level='O2', loss_scale_manager=loss_scale,
                      keep_batchnorm_fp32=False, metrics={'acc'}, frequency=config.frequency)
    else:
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                      amp_level="O2", keep_batchnorm_fp32=True, frequency=config.frequency)
    ...
```

### Running the Script

After the training script is defined, call the shell script in the `scripts` directory to start the distributed training process.

#### Ascend 910

Currently, MindSpore distributed execution on Ascend uses the single-device single-process running mode. That is, one process runs on one device, and the number of total processes is the same as the number of devices that are being used. All processes are executed in the background. Create a directory named `train_parallel`+`device_id` for each process to store log information, operator compilation information, and training checkpoint files. The following takes the distributed training script for eight devices as an example to describe how to run the script:

Run the script.

```bash
sh run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [DEVICE_NUM]
```

Variables `RANK_TABLE_FILE`, `DATASET_PATH`, and `DEVICE_NUM` need to be transferred to the script. The meanings of variables are as follows:

- `RANK_TABLE_FILE`: path for storing the networking information file (about the rank table file, you can refer to [HCCL_TOOL](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/utils/hccl_tools))
- `DATASET_PATH`: training dataset path
- `DEVICE_NUM`: the actual number of running devices.

For details about other environment variables, see configuration items in the installation guide.

The following is an example of loss values output during training:

```bash
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

```bash
└─train_parallel0
    ├─resnet-1_5004.ckpt
    ├─resnet-2_5004.ckpt
    │      ......
    ├─resnet-42_5004.ckpt
    │      ......
```

In the preceding information,
`*.ckpt` indicates the saved model parameter file. The name of a checkpoint file is in the following format: *Network name*-*Number of epochs*_*Number of steps*.ckpt.

#### GPU

On the GPU hardware platform, MindSpore uses `mpirun` of OpenMPI to perform distributed training. The process creates a directory named `train_parallel` to store log information and training checkpoint files. The following takes the distributed training script for eight devices as an example to describe how to run the script:

```bash
sh run_distribute_train_gpu.sh [DATASET_PATH] [DEVICE_NUM]
```

Variables `DATASET_PATH` and `DEVICE_NUM` need to be transferred to the script. The meanings of variables are as follows:

- `DATASET_PATH`: training dataset path
- `DEVICE_NUM`: the actual number of running devices

During GPU-based training, the `DEVICE_ID` environment variable is not required. Therefore, you do not need to call `int(os.getenv('DEVICE_ID'))` in the main training script to obtain the device ID or transfer `device_id` to `context`. You need to set `device_target` to `GPU` and call `init()` to enable the NCCL.

The following is an example of loss values output during training:

```bash
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

```bash
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
    net.add_flags_recursive(thor=False)

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    keys = list(param_dict.keys())
    for key in keys:
        if "damping" in key:
            param_dict.pop(key)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
```

### Inference

After the inference network is defined, the shell script in the `scripts` directory is called for inference.

#### Ascend 910

On the Ascend 910 hardware platform, run the following inference command:

```bash
sh run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

Variables `DATASET_PATH` and `CHECKPOINT_PATH` need to be transferred to the script. The meanings of variables are as follows:

- `DATASET_PATH`: inference dataset path
- `CHECKPOINT_PATH`: path for storing the checkpoint file

Currently, a single device (device 0 by default) is used for inference. The inference result is as follows:

```text
result: {'top_5_accuracy': 0.9295574583866837, 'top_1_accuracy': 0.761443661971831} ckpt=train_parallel0/resnet-42_5004.ckpt
```

- `top_5_accuracy`: For an input image, if the labels whose prediction probability ranks top 5 match actual labels, the classification is correct.
- `top_1_accuracy`: For an input image, if the label with the highest prediction probability is the same as the actual label, the classification is correct.

#### GPU

On the GPU hardware platform, run the following inference command:

```bash
sh run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

Variables `DATASET_PATH` and `CHECKPOINT_PATH` need to be transferred to the script. The meanings of variables are as follows:

- `DATASET_PATH`: inference dataset path
- `CHECKPOINT_PATH`: path for storing the checkpoint file

The inference result is as follows:

```text
result: {'top_5_accuracy': 0.9287972151088348, 'top_1_accuracy': 0.7597031049935979} ckpt=train_parallel/resnet-36_5004.ckpt
```
