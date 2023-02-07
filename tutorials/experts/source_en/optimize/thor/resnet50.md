# Applying Second-Order Optimization Practices on the ResNet-50 Network

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/experts/source_en/optimize/thor/resnet50.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

## Overview

Common optimization algorithms can be divided into first-order optimization algorithms and second-order optimization algorithms. Classical first-order optimization algorithms, such as SGD, have small volume of computation, with fast computation speed, but converge slowly and require many iterations. The second-order optimization algorithms use the second-order derivative of the objective function to accelerate the convergence, which can converge to the optimal value of the model faster and requires fewer iterations. However, the overall execution time of the second-order optimization algorithms is still slower than the first-order optimization algorithms due to its high computational cost, so the application of the second-order optimization algorithm in deep neural network training is not common at present. The main computational cost of the second-order optimization algorithms lies in the inverse operation of the second-order information matrix (Hessian matrix, [FIM matrix](https://arxiv.org/pdf/1808.07172.pdf), etc.), with a time complexity of about $O(n^3)$.

Based on the existing natural gradient algorithm, the MindSpore development team has developed a usable second-order optimizer THOR by using approximations, tiles and other optimization accelerations for the FIM matrix, which greatly reduces the computational complexity of the inverse matrix. Using eight Ascend 910 AI processors, THOR can complete training the ResNet50-v1.5 network and ImageNet dataset in 72min, nearly doubling the speed compared to SGD+Momentum.

This tutorial will focus on how to train ResNet50-v1.5 network and ImageNet dataset on Ascend 910 and GPU using THOR, a second-order optimizer provided by MindSpore.
> Download the complete sample code: [Resnet](https://gitee.com/mindspore/models/tree/r2.0.0-alpha/official/cv/ResNet).

The directory structure of sample code:

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

1. Prepare ImageNet datasets and process the required datasets.
2. Define the ResNet50 network.
3. Define the loss function and THOR optimizer.
4. Load the dataset and train it, and view the results and save the model file after the training is completed.
5. Load the saved model for inference.

## Preparation

Make sure MindSpore is properly installed before practicing. If not, you can install MindSpore through the [MindSpore installation page](https://www.mindspore.cn/install).

### Preparing the Dataset

Download the complete ImageNet2012 dataset and unzip the dataset to `ImageNet2012/ilsvrc` and `ImageNet2012/ilsvrc_eval` paths in the local workspace respectively.

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

Refer to [Distributed Parallel Training (Ascend)](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/train_ascend.html#configuring-distributed-environment-variables) for the configuration of distributed environment variables for the Ascend 910 AI processor.

#### GPU

Refer to [Distributed Parallel Training (GPU)](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/train_gpu.html#configuring-distributed-environment) for the configuration of distributed environment variables for the GPU.

## Loading and Processing the Datasets

For distributed training, the dataset is loaded in a parallel manner, while the dataset is processed through the data augmentation interface provided by MindSpore. The script to load and process the datasets is in the `src/dataset.py` script in the source code.

```python
import os
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
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
            vision.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]
    else:
        trans = [
            vision.Decode(),
            vision.Resize(256),
            vision.CenterCrop(image_size),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]

    type_cast_op = transforms.TypeCast(ms.int32)

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

> MindSpore supports a variety of data processing and augmentation operations, often in combination, as described in the [Data Processing](https://www.mindspore.cn/tutorials/en/r2.0.0-alpha/advanced/dataset.html) and [Data Augmentation](https://www.mindspore.cn/tutorials/en/r2.0.0-alpha/advanced/dataset.html) chapters.

## Defining the Networks

The network model used in this example is ResNet50-v1.5, defining the [ResNet50 network](https://gitee.com/mindspore/models/blob/r2.0.0-alpha/official/cv/ResNet/src/resnet.py).

After the network is constructed, the defined ResNet50 is called in the `__main__` function.

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

## Defining the Loss Function and THOR Optimizer

### Defining the Loss Function

The loss functions supported by MindSpore are `SoftmaxCrossEntropyWithLogits`, `L1Loss`, `MSELoss`, etc. The THOR optimizer requires the `SoftmaxCrossEntropyWithLogits` loss function.

The steps to implement the loss function are in the `src/CrossEntropySmooth.py` script. A common trick in deep network model training is used here: label smoothing, which can increase the generalization ability of the model by smoothing the real labels and improving the tolerance of the model to misclassified labels.

```python
class CrossEntropySmooth(LossBase):
    """CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = ops.OneHot()
        self.sparse = sparse
        self.on_value = ms.Tensor(1.0 - smooth_factor, ms.float32)
        self.off_value = ms.Tensor(1.0 * smooth_factor / (num_classes - 1), ms.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss
```

Call the defined loss function in the `__main__` function:

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

### Defining the Optimizers

The parameter update formula for the THOR optimizer is as follows:

$$ \theta^{t+1} = \theta^t + \alpha F^{-1}\nabla E$$

The meaning of each parameter in the parameter update formula is as follows:

- $\theta$: Trainable parameters in the network.
- $t$: The number of iterations.
- $\alpha$: The learning rate value, the update step of the parameter.
- $F^{-1}$: FIM matrix, obtained by calculation in the network.
- $\nabla E$: First-order gradient values.

As can be seen from the parameter update formula, what the THOR optimizer needs to calculate additionally is the FIM matrix for each layer. The FIM matrix can be adaptively adjusted to the step size and direction in each layer of parameter updates, and reduce the complexity of parameters tuning while accelerating the convergence.

For more detailed introduction to THOR optimizer, refer to [THOR article](https://www.aaai.org/AAAI21Papers/AAAI-6611.ChenM.pdf).

When calling the MindSpore-encapsulated second-order optimizer THOR, the optimizer automatically calls the transformation interface to convert the Conv2d layer and Dense layer in the previously defined ResNet50 network into the corresponding [Conv2dThor](https://gitee.com/mindspore/mindspore/blob/r2.0.0-alpha/mindspore/python/mindspore/nn/layer/thor_layer.py) and [DenseThor](https://gitee.com/mindspore/mindspore/blob/r2.0.0-alpha/mindspore/python/mindspore/nn/layer/thor_layer.py).
And the computation and storage of the second-order information matrix can be done in Conv2dThor and DenseThor.

> The network backbone is the same before and after the THOR optimizer conversion, and the network parameters remain unchanged.

Calling the THOR optimizer in the training master script:

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
    opt = thor(net, ms.Tensor(lr), ms.Tensor(damping), config.momentum, config.weight_decay, config.loss_scale,
               config.batch_size, split_indices=split_indices, frequency=config.frequency)
    ...
```

## Training the Networks

### Configuring Model Saving

MindSpore provides a callback mechanism to execute custom logic during training, here using the `ModelCheckpoint` function provided by the framework.
`ModelCheckpoint` can save the network model and parameters for subsequent fine-tuning operations.
`TimeMonitor`, `LossMonitor` are the official callback functions provided by MindSpore, which can be used to monitor the changes of single-step iteration time and `loss` values during the training process respectively.

```python
...
import mindspore as ms
from mindspore.train import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
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

### Configuring the Training Network

Training of the network can be easily performed through the `model.train` interface provided by MindSpore. The THOR optimizer reduces the volume of computation and improves the computation speed by reducing the frequency of second-order matrix updates, so it redefines a [ModelThor](https://gitee.com/mindspore/mindspore/blob/r2.0.0-alpha/mindspore/python/mindspore/train/train_thor/model_thor.py) class and inherits the Model class provided by MindSpore. Obtaining the second-order matrix update frequency control parameter of THOR in the ModelThor class, users can optimize the overall performance by adjusting this parameter.
MindSpore provides a one-click conversion interface from Model class to ModelThor class.

```python
...
import mindspore as ms
from mindspore.train import Model
...

if __name__ == "__main__":
    ...
    loss_scale = ms.FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                  amp_level="O2", keep_batchnorm_fp32=False, eval_network=dist_eval_network)
    if cfg.optimizer == "Thor":
        model = ms.ConvertModelUtils().convert_to_thor_model(model=model, network=net, loss_fn=loss, optimizer=opt,
                                                          loss_scale_manager=loss_scale, metrics={'acc'},
                                                          amp_level="O2", keep_batchnorm_fp32=False)  
    ...
```

### Running the Script

After the training script is defined, call the shell script in the `scripts` directory and start the distributed training process.

#### Ascend 910

The current MindSpore distributed executes in the running mode of single-card, single-process on Ascend, i.e., 1 process running on each card, with the number of processes matching the number of used cards. The processes are executed in the background and each process creates a directory called `train_parallel` + `device_id` to store log information, operator compilation information and training checkpoint files. The following is an example of a distributed training script by using 8 cards to demonstrate how to run the script.

Use the following command to run the script:

```bash
bash run_distribute_train.sh <RANK_TABLE_FILE> <DATASET_PATH> [CONFIG_PATH]
```

The script needs to pass in the variables `RANK_TABLE_FILE`, `DATASET_PATH` and `CONFIG_PATH`, where:

- `RANK_TABLE_FILE`: The path of networking information file. (For the generation of rank table files, refer to [HCCL_TOOL](https://gitee.com/mindspore/models/tree/r2.0.0-alpha/utils/hccl_tools).)
- `DATASET_PATH`: The path of the training dataset.
- `CONFIG_PATH`: The path of configuration file.

For the rest of the environment variables, please refer to the configuration items in the installation tutorial.

An example of loss printing during training is as follows:

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

After training, the checkpoint file generated by each card training is saved in the respective training directory. An example of the checkpoint file generated by `device_0` is as follows:

```text
└─train_parallel0
    ├─ckpt_0
        ├─resnet-1_5004.ckpt
        ├─resnet-2_5004.ckpt
        │      ......
        ├─resnet-42_5004.ckpt
        │      ......
```

where
`*.ckpt` refers to the saved model parameter file. The specific meanings of checkpoint file names: *network name*-*number of epoch*_*number of step*.ckpt.

#### GPU

On the GPU hardware platform, MindSpore uses OpenMPI `mpirun` for distributed training. The process creates a directory called `train_parallel` to store log information and checkpoint files for training. The following is an example of a distributed training script using 8 cards to demonstrate how to run the script.

Use the following command to run the script:

```bash
bash run_distribute_train_gpu.sh <DATASET_PATH> <CONFIG_PATH>
```

The script needs to pass in the variables `DATASET_PATH` and `CONFIG_PATH`, where

- `DATASET_PATH`: Training dataset path.
- `CONFIG_PATH`: Configuration file path.

During GPU training, there is no need to set the `DEVICE_ID` environment variable. So there is no need to call `int(os.getenv('DEVICE_ID'))` to get the physical serial number of the card in the main training script, and there is no need to pass `device_id` in the `context`. We need to set device_target to GPU and call `init()` to enable NCCL.

An example of loss printing during training is as follows:

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

After training, an example of the saved model file is as follows:

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

Inference is performed by using the checkpoint file saved during the training process to verify the generalization ability of the model. First load the model file through the `load_checkpoint` interface, call the `eval` interface of `Model` to make a prediction on the input image category, and then compare it with the real category of the input image, to get the final prediction accuracy value.

### Defining the Inference Network

1. Use the `load_checkpoint` interface to load the model file.
2. Use the `model.eval` interface to read in the test dataset for inference.
3. Calculate the prediction accuracy value.

```python
...
import mindspore as ms
from mindspore.train import Model
...

if __name__ == "__main__":
    ...
    # define net
    net = resnet(class_num=config.class_num)

    # load checkpoint
    param_dict = ms.load_checkpoint(args_opt.checkpoint_path)
    ms.load_param_into_net(net, param_dict)
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

### Executing the Inference

After the inference network is defined, the shell script in the `scripts` directory is called for inference.

#### Ascend 910

On the Ascend 910 hardware platform, the inference execution command is as follows:

```bash
bash run_eval.sh <DATASET_PATH> <CHECKPOINT_PATH> <CONFIG_PATH>
```

The script needs to pass in the variables `DATASET_PATH`, `CHECKPOINT_PATH` and `<CONFIG_PATH>`, where

- `DATASET_PATH`: The inference dataset path.
- `CHECKPOINT_PATH`: The saved checkpoint path.
- `CONFIG_PATH`: The configuration file path.

The current inference is performed using a single card (default device 0), and the result of the inference is as follows:

```text
result: {'top_5_accuracy': 0.9295574583866837, 'top_1_accuracy': 0.761443661971831} ckpt=train_parallel0/resnet-42_5004.ckpt
```

- `top_5_accuracy`: For an input image, a classification is considered correct if the top five tags in the predicted probability ranking contain true tags.
- `top_1_accuracy`: For an input image, if the label with the highest predicted probability is the same as the true label, the classification is considered correct.

#### GPU

On the GPU hardware platform, the execution command for inference is as follows:

```bash
  bash run_eval_gpu.sh <DATASET_PATH> <CHECKPOINT_PATH> <CONFIG_PATH>
```

The script needs to pass in the variables `DATASET_PATH`, `CHECKPOINT_PATH` and `CONFIG_PATH`, where

- `DATASET_PATH`: The inference dataset path.
- `CHECKPOINT_PATH`: The saved checkpoint path.
- `CONFIG_PATH`: The configuration file path.

The inference result is as follows:

```text
result: {'top_5_accuracy': 0.9287972151088348, 'top_1_accuracy': 0.7597031049935979} ckpt=train_parallel/resnet-36_5004.ckpt
```
