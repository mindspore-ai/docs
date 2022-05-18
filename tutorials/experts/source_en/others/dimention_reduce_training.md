# Dimension Reduction Training Algorithm

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/others/dimention_reduce_training.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## Overview

This tutorial introduces the training method of dimension reduction training, and the purpose is to solve the problem of slow convergence of the network in the later stage of training.

In general network training, the early convergence is relatively fast. The late convergence enters the stable stage, and the convergence is slower. In order to improve the convergence speed in the later stage, the dimension reduction training divides the network training into two stages. In the first stage, the network is trained in the traditional way, keeping N (N>32) weight files. It is recommended to abandon the weight files in the early stage, such as the first stage of training 50 epochs, starting from the 21st epoch to save the weight file. Each epoch saves 2 weight files, and there are 60 weight files at the end of the first stage. In the second stage, the weight file obtained in the first stage is loaded, and the PCA (Principal Component Analysis) is reduced. The weight is reduced from high dimension (M) to low dimension (32). The gradient descent direction and length of the weight are searched for in the low dimension, and then the weight is backprojected to the high dimension, updating the weight.

In view of the time-consuming  single-card training, this tutorial will use the data parallel mode on the Ascend 910 AI processor hardware platform to introduce the regular training of the first stage and how to achieve the second phase of dimension reduction training in the Boost mode by taking the resNet-50 training process on ImageNet 2012 as an example.

## Preparation

>Download the complete sample code from:
>
><https://gitee.com/mindspore/docs/tree/master/docs/sample_code/dimension_reduce_training>
>
>The models library links referenced in the code:
>
><https://gitee.com/mindspore/models>

### Configuring the Distributed Environment Variables

When performing distributed training on the local Ascend processor, you need to configure the networking information file of the current multi-card environment. The json file of an 8-card environment is configured as follows, and the configuration file is named rank_table_8pcs.json. rank_table can be generated using the [hccl_tools.py](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/hccl_tools.py) below the models.

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "10.155.111.140",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

### Preparing the Dataset

Used dataset: [ImageNet 2012](http://www.image-net.org/)

- The size of the dataset: 1000 classes and 224*224 color images in totoal

    - Training set: 1,281,167 images in total

    - Test set: 50000 images in total

- Data format: JPEG
- Download the dataset, and the directory structure is as follows:

```text
└─dataset
    ├─train                 # Training the dataset
    └─validation_preprocess # Evaluating the dataset
```

## First Stage: Regular Training

> The main training sample code for the first phase:
>
> <https://gitee.com/mindspore/docs/blob/master/docs/sample_code/dimension_reduce_training/train_stage_1.py>.

### Runtime Mode and Backend Device Settings

Specify the operating mode, running card number, parallel mode. through the context interface provided by MindSpore, and initialize the HCCL communication through init.

```python
set_context(mode=GRAPH_MODE, device_target=args.device_target)
device_id = int(os.getenv('DEVICE_ID'))
set_context(device_id=device_id)
set_auto_parallel_context(device_num=8, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
set_algo_parameters(elementwise_op_strategy_follow=True)
all_reduce_fusion_config = [85, 160]
set_auto_parallel_context(all_reduce_fusion_config=all_reduce_fusion_config)
init()
```

### Loading the Dataset

Image loading interface ImageFolderDataset is used to load the ImageNet 2012 dataset by using MindSpore. The dataset is processed through the data augmentation interface provided by MindSpore, and this part of the code is imported by [dataset.py](https://gitee.com/mindspore/models/blob/master/official/cv/resnet/src/dataset.py) in the `resnet` directory in the models.

```python
# define train dataset
train_data_path = os.path.join(args.data_path, "train")
ds_train = create_dataset(dataset_path=train_data_path, do_train=True, batch_size=256, train_image_size=224,
                          eval_image_size=224, target=args.device_target, distribute=True)
step_size = ds_train.get_dataset_size()
```

### Defining the Network

The build code for the ResNet-50 network is imported by [resnet.py](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/adasum/resnet.py).

```python
# define net
net = resnet(class_num=1001)
init_weight(net=net)
```

### Defining the Training Model

Define the loss functions loss and optimizer required by the model.

Loss uses CrossEntropySmooth, imported by [CrossEntropySmooth.py](https://gitee.com/mindspore/models/blob/master/official/cv/resnet/src/CrossEntropySmooth.py) in the `resnet` directory in ModelZoo.

The build code for the learning rate lr is imported by [lr_generator.py](https://gitee.com/mindspore/models/blob/master/official/cv/resnet/src/lr_generator.py) in the `resnet` directory in the models.

```python
# define loss
loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, num_classes=1001)
loss_scale = FixedLossScaleManager(1024, drop_overflow_update=False)

# define optimizer
group_params = init_group_params(net)
lr = get_lr(lr_init=0, lr_end=0.0, lr_max=0.8, warmup_epochs=5, total_epochs=90, steps_per_epoch=step_size,
            lr_decay_mode="linear")
lr = Tensor(lr)
opt = Momentum(group_params, lr, 0.9, loss_scale=1024)

# define metrics
metrics = {"acc"}

# define model
model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics, amp_level="O2",
              boost_level="O0", keep_batchnorm_fp32=False)
```

### Training the Model

Before the training starts, define the callback function callback, add the training time information output, loss information output, and weight saving, where the model weights are only saved on 0 cards.

callback_1 save all the weights of the network and the state of the optimizer, and callback_2 only the weights in the network that participate in the update for the second phase of PCA.

```python
# define callback_1
cb = [TimeMonitor(data_size=step_size), LossMonitor()]
if get_rank_id() == 0:
    config_ck = CheckpointConfig(save_checkpoint_steps=step_size * 10, keep_checkpoint_max=10)
    ck_cb = ModelCheckpoint(prefix="resnet", directory="./checkpoint_stage_1", config=config_ck)
    cb += [ck_cb]

# define callback_2: save weights for stage 2
if get_rank_id() == 0:
    config_ck = CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=40,
                                 saved_network=net)
    ck_cb = ModelCheckpoint(prefix="resnet", directory="./checkpoint_stage_1/checkpoint_pca", config=config_ck)
    cb += [ck_cb]

print("============== Starting Training ==============")
model.train(70, ds_train, callbacks=cb, sink_size=step_size, dataset_sink_mode=True)
```

### Testing the Model

First define the test model, then load the test data set to test the accuracy of the model. Judged by rank_id, the model is tested only on 0 cards.

```python
if get_rank_id() == 0:
    print("============== Starting Testing ==============")
    eval_data_path = os.path.join(args.data_path, "val")
    ds_eval = create_dataset(dataset_path=eval_data_path, do_train=False, batch_size=256, target="Ascend")
    if ds_eval.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))
```

### Experiment Result

After 70 rounds of epoch, the accuracy on the test set is about 66.05%.

1. Call the run script [run_stage_1.sh] (https://gitee.com/mindspore/docs/blob/master/docs/sample_code/dimension_reduce_training/run_stage_1.sh) to view the run results. Running the script requires a given dataset path, and the model is saved under device0_stage_1/checkpoint_stage_1 by default.

   ```bash
   bash run_stage_1.sh ./imagenet
   ```

   The output is as follows, and you can see that the loss value gradually decreases with the training:

   ```text
   ============== Starting Training ==============
   epoch: 1 step: 625 loss is  5.2477064
   ...
   epoch: 10 step: 625 loss is  3.0178385
   ...
   epoch: 30 step: 625 loss is  2.4231198
   ...
   ...
   epoch: 70 step: 625 loss is  2.3120291
   ```

2. View the inference precision.

   ```text
   ============== Starting Testing ==============
   ============== {'Accuracy': 0.6604992988782051} ==============
   ```

## Second Stage: boost Mode

Based on the weight file obtained in the first stage, in the Boost mode, we can realize the function of dimension reduction training by simply calling the dimension reduction training interface of mindspore.boost.

> The main training sample code for the second phase:
>
> <https://gitee.com/mindspore/docs/blob/master/docs/sample_code/dimension_reduce_training/train_boost_stage_2.py>.

The code for the second and first stages is essentially the same, and only the inconsistencies are described below.

### Defining the network and loading the initialization weights

After the network is built, the weight file at the end of the first stage of training is loaded, and the second stage of training is carried out on this basis.

```python
# define net
net = resnet(class_num=1001)
if os.path.isfile(args.pretrained_weight_path):
    weight_dict = load_checkpoint(args.pretrained_weight_path)
load_param_into_net(net, weight_dict)
```

### Defining the Training Model

Unlike the first phase, the second stage uses the SGD optimizer to update the network weights. In addition to this, the parameters required for dimension reduction training need to be configured, which is built in the form of a dictionary. First of all, you need to set the type of boost to manual setting, open dimension reduction training (dim_reduce), then configure the number of cards used by each server node, and finally configure some hyperparameters of dimension reduction training. In addition, when defining the model, you also need to open the boost_level (set to "O1" or "O2").

```python
# define loss
loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, num_classes=1001)

# define optimizer
group_params = init_group_params(net)
opt = SGD(group_params, learning_rate=1)

# define metrics
metrics = {"acc"}

# define boost config dictionary
boost_dict = {
    "boost": {
        "mode": "manual",
        "dim_reduce": True
    },
    "common": {
        "device_num": 8
    },
    "dim_reduce": {
        "rho": 0.55,
        "gamma": 0.9,
        "alpha": 0.001,
        "sigma": 0.4,
        "n_component": 32,                                                 # PCA component
        "pca_mat_path": args.pca_mat_path,                                 # the path to load pca mat
        "weight_load_dir": "./checkpoint_stage_1/checkpoint_pca",          # the directory to load weight file saved as ckpt.
        "timeout": 1200
    }
}

# define model
model = Model(net, loss_fn=loss, optimizer=opt, metrics=metrics, boost_level="O1", boost_config_dict=boost_dict)
```

### Training the Model

In the second stage, we set epoch as 2.

```python
# define callback
cb = [TimeMonitor(data_size=step_size), LossMonitor()]
if get_rank_id() == 0:
    config_ck = CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=2)
    ck_cb = ModelCheckpoint(prefix="resnet", directory="./checkpoint_stage_2", config=config_ck)
    cb += [ck_cb]

print("============== Starting Training ==============")
model.train(2, ds_train, callbacks=cb, sink_size=step_size, dataset_sink_mode=True)
```

### Experiment Result

After 2 rounds of epoch, the accuracy on the test set is about 74.31%.

1. Call the run script [run_stage_2.sh](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/dimension_reduce_training/run_stage_2.sh) to view the run results. Running the script requires a given dataset path. Weight file at the end of the first phase of training, and weight file saved in the second phase under device0_stage_2/checkpoint_stage_2 by default.

   ```bash
   bash run_stage_2.sh ./imagenet ./device0_stage_1/checkpoint_stage_1/resnet-70_625.ckpt
   ```

   If you have already done PCA to the weights of the first stage and saved the feature transformation matrix, you can give the feature transformation matrix file path, eliminating the process of finding the feature transformation matrix.

   ```bash
   bash run_stage_2.sh ./imagenet ./device0_stage_1/checkpoint_stage_1/resnet-70_625.ckpt /path/pca_mat.npy
   ```

   The output is as follows, and you can see that the loss value gradually decreases with the training:

   ```text
   epoch: 1 step: 625 loss is  2.3422508
   epoch: 2 step: 625 loss is  2.1641185
   ```

2. Look at the inference precision, and the code saves the checkpoint to the current directory, and then loads the checkpoint to perform the inference.

   ```text
   ============== Starting Testing ==============
   ============== {'Accuracy': 0.7430964543269231} ==============
   ```

Generally ResNet-50 training 80 epochs also only reached 70.18%. Using dimension reduction training, we can reach 74.31% in 72 epochs.