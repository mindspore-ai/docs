# Using MobileNetV2 to Implement Fine-Tuning

`Ascend` `GPU` `CPU` `Whole Peocess`

<!-- TOC -->

- [Using MobileNetV2 to Implement Fine-Tuning](#using-mobilenetv2-to-implement-fine-tuning)
    - [Overview](#overview)
    - [Task Description and Preparations](#task-description-and-preparations)
        - [Environment Configuration](#environment-configuration)
        - [Downloading Code](#downloading-code)
        - [Preparing a Pre-Trained Model](#preparing-a-pre-trained-model)
        - [Preparing Data](#preparing-data)
    - [Code for Loading a Pre-Trained Model](#code-for-loading-a-pre-trained-model)
    - [Parameter Description](#parameter-description)
        - [Running Python Files](#running-python-files)
        - [Running Shell Scripts](#running-shell-scripts)
    - [Loading Fine-Tuning Training](#loading-fine-tuning-training)
        - [Loading Training on CPU](#loading-training-on-cpu)
        - [Loading Training on GPU](#loading-training-on-gpu)
        - [Loading Training on Ascend AI Processor](#loading-training-on-ascend-ai-processor)
        - [Fine-Tuning Training Result](#fine-tuning-training-result)
    - [Validating the Fine-Tuning Training Model](#validating-the-fine-tuning-training-model)
        - [Validating the Model](#validating-the-model)
        - [Validation Result](#validation-result)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_en/cv_mobilenetv2_fine_tune.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

## Overview

In a computer vision task, training a network from scratch is time-consuming and requires a large amount of computing power. Pre-trained models often select open large datasets such as OpenImage, ImageNet, VOC, and COCO. The number of images in these datasets reaches hundreds of thousands or even millions. Most tasks have a large amount of data. If a pre-trained model is not used during network model training, the training from scratch consumes a large amount of time and computing power. As a result, the model is prone to local minimum and overfitting. Therefore, most tasks perform fine-tuning on pre-trained models.

MindSpore is a diversified machine learning framework. It can run on devices such as mobile phones and PCs, or on server clusters on the cloud. Currently, MobileNetV2 supports fine-tuning on a single CPU or on one or more Ascend AI Processors or GPUs on Windows, EulerOS, and Ubuntu systems. This tutorial describes how to perform fine-tuning training and validation in the MindSpore frameworks of different systems and processors.

Currently, only the CPU is supported on Windows, and the CPU, GPU, and Ascend AI Processor are supported on Ubuntu and EulerOS.

> You can obtain the complete executable sample code at <https://gitee.com/mindspore/models/tree/r1.5/official/cv/mobilenetv2>.

## Task Description and Preparations

### Environment Configuration

If running a task in a local environment, install the MindSpore framework and configure the CPU, GPU, or Ascend AI Processor. If running a task in the HUAWEI CLOUD environment, skip this section because the installation and configuration are not required.

On the Windows operating system, backslashes `\` are used to separate directories of different levels in a path address. On the Linux operating system, slashes `/` are used. The following uses `/` by default. If you use Windows operating system, replace `/` in the path address with `\`.

1. Install the MindSpore framework.
    [Install](https://www.mindspore.cn/install/en) a MindSpore framework based on the processor architecture and the EulerOS, Ubuntu, or Windows system.

2. Configure the CPU environment.  
    Set the following code before calling the CPU to start training or testing:

    ```python
    if config.platform == "CPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, \
            save_graphs=False)
    ```

3. Configure the GPU environment.  
    Set the following code before calling the GPU to start training or testing:

    ```python
    elif config.platform == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, save_graphs=False)
        if config.run_distribute:
            init("nccl")
            context.set_auto_parallel_context(device_num=get_group_size(),
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    ```

4. Configure the Ascend environment.  
    The following uses the JSON configuration file `hccl_config.json` in an environment with eight Ascend 910 AI processors as an example. Adjust `"server_count"` and `device` based on the following example to switch between the single-device and multi-device environments:

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

    Set the following code before calling the Ascend AI Processor to start training or testing:

    ```python
    elif config.platform == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, device_id=config.device_id,
                            save_graphs=False)
        if config.run_distribute:
            context.set_auto_parallel_context(device_num=config.rank_size,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              all_reduce_fusion_config=[140])
            init()
    ...
    ```

### Downloading Code

Run the following command to clone [MindSpore open-source project repository](https://gitee.com/mindspore/mindspore.git) in Gitee and go to `./model_zoo/official/cv/mobilenetv2/`.

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.5
cd ./mindspore/model_zoo/official/cv/mobilenetv2
```

The code structure is as follows:

```text
├── MobileNetV2
  ├── README.md                  # descriptions about MobileNetV2
  ├── ascend310_infer            # application for 310 inference
  ├── scripts
  │   ├──run_infer_310.sh        # shell script for 310 inference
  │   ├──run_train.sh            # shell script for training, fine-tuning or incremental learning with CPU, GPU or Ascend
  │   ├──run_eval.sh             # shell script for evaluation with CPU, GPU or Ascend
  │   ├──cache_util.sh           # a collection of helper functions to manage cache
  │   ├──run_train_nfs_cache.sh  # shell script for training with NFS dataset and leverage caching service for better performance
  ├── src
  │   ├──aipp.cfg                # aipp config
  │   ├──dataset.py              # creating dataset
  │   ├──lr_generator.py         # learning rate config
  │   ├──mobilenetV2.py          # MobileNetV2 architecture
  │   ├──models.py               # loading define_net, Loss and Monitor
  │   ├──utils.py                # loading ckpt_file for fine-tuning or incremental learning
  │   └──model_utils
  │      ├──config.py            # processing configuration parameters
  │      ├──device_adapter.py    # getting cloud ID
  │      ├──local_adapter.py     # getting local ID
  │      └──moxing_adapter.py    # parameter processing
  ├── default_config.yaml        # training parameter profile(ascend)
  ├── default_config_cpu.yaml    # training parameter profile(cpu)
  ├── default_config_gpu.yaml    # training parameter profile(gpu)
  ├── train.py                   # training script
  ├── eval.py                    # evaluation script
  ├── export.py                  # exporting mindir script
  ├── mindspore_hub_conf.py      # mindspore hub interface
  ├── postprocess.py             # postprocess script
```

During fine-tuning training and testing, python files `train.py` and `eval.py` can be used on Windows, Ubuntu, and EulerOS, and shell script files `run_train.sh` and `run_eval.sh` can be used on Ubuntu and EulerOS.

If the script file `run_train.sh` is used, it runs `launch.py` and inputs parameters to `launch.py` which starts one or more processes to run `train.py` based on the number of allocated CPUs, GPUs, or Ascend AI Processors. Each process is allocated with a processor.

### Preparing a Pre-Trained Model

Download a [CPU/GPU pre-trained model](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2_cpu_gpu.ckpt) or [Ascend pre-trained model](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2_ascend.ckpt) to the following directories based on the processor type:  
`./pretrain_checkpoint/`

- CPU/GPU

    ```bash
    mkdir pretrain_checkpoint
    wget -P ./pretrain_checkpoint https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2_cpu_gpu.ckpt --no-check-certificate
    ```

- Ascend AI Processor

    ```bash
    mkdir pretrain_checkpoint
    wget -P ./pretrain_checkpoint https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2_ascend.ckpt --no-check-certificate
    ```

### Preparing Data

Prepare the dataset managed in ImageFolder format. Add the `<dataset_path>` parameter when running `run_train.sh`, and add the `--dataset_path <dataset_path>` parameter when running `train.py`.

The dataset structure is as follows:

```text
└─ImageFolder
    ├─train
    │   class1Folder
    │   class2Folder
    │   ......
    └─eval
        class1Folder
        class2Folder
        ......
```

## Code for Loading a Pre-Trained Model

During fine-tuning, you need to load a pre-trained model. The distribution of the feature extraction layer (convolutional layer) in different datasets and tasks tends to be consistent. However, the combination of feature vectors (fully connected layer) is different, and the number of classes (output_size of the fully connected layer) is usually different. During fine-tuning, parameters of the feature extraction layer are loaded and trained, while those of the fully connected layer are not. During fine-tuning and initial training, both feature extraction layer parameters and fully connected layer parameters are loaded and trained.

Before training and testing, build a backbone network and a head network of MobileNetV2 on the first line of the code, and build a MobileNetV2 network containing the two subnets. Lines 3 to 10 of the code show how to define `backbone_net` and `head_net` and how to add the two subnets to `mobilenet_v2`. Lines 12 to 27 of the code show that in fine-tuning training mode, the pre-trained model needs to be loaded to the `backbone_net` subnet, and parameters in `backbone_net` are frozen and do not participate in training. Lines 21 to 27 of the code show how to freeze network parameters.

```python
 1:  backbone_net, head_net, net = define_net(config, config.is_training)
 2:  ...
 3:  def define_net(config, is_training=True):
 4:      backbone_net = MobileNetV2Backbone()
 5:      activation = config.activation if not is_training else "None"
 6:      head_net = MobileNetV2Head(input_channel=backbone_net.out_channels,
 7:                              num_classes=config.num_classes,
 8:                              activation=activation)
 9:      net = mobilenet_v2(backbone_net, head_net)
10:      return backbone_net, head_net, net
11:  ...
12:  if config.pretrain_ckpt:
13:      if config.freeze_layer == "backbone":
14:         load_ckpt(backbone_net, config.pretrain_ckpt, trainable=False)
15:         step_size = extract_features(backbone_net, config.dataset_path, config)
16:      elif config.filter_head:
17:           load_ckpt(backbone_net, config.pretrain_ckpt)
18:      else:
19:           load_ckpt(net, config.pretrain_ckpt)
20:  ...
21:  def load_ckpt(network, pretrain_ckpt_path, trainable=True):
22:      """ train the param weight or not """
23:      param_dict = load_checkpoint(pretrain_ckpt_path)
24:      load_param_into_net(network, param_dict)
25:      if not trainable:
26:          for param in network.get_parameters():
27:              param.requires_grad = False
```

## Parameter Description

Change the value of each parameter based on the local processor type, data path, and pre-trained model path.

### Running Python Files

When using `train.py` for training on Windows and Linux, input `config_path`, `dataset_path`, `platform`, `pretrain_ckpt`, and `freeze_layer`. When using `eval.py` for validation, input `config_path`, `dataset_path`, `platform`, and `pretrain_ckpt`.

```bash
# Windows/Linux train with Python file
python train.py --config_path [CONFIG_PATH] --platform [PLATFORM] --dataset_path <DATASET_PATH>  --pretrain_ckpt [PRETRAIN_CHECKPOINT_PATH] --freeze_layer[("none", "backbone")]

# Windows/Linux eval with Python file
python eval.py --config_path [CONFIG_PATH] --platform [PLATFORM] --dataset_path <DATASET_PATH> --pretrain_ckpt <PRETRAIN_CHECKPOINT_PATH>
```

- `--config_path`: parameters required for training and verification.
- `--dataset_path`: path of the training or validation dataset. There is no default value. This parameter is mandatory for training or validation.
- `--platform`: processor type. The default value is `Ascend`. You can set it to `CPU` or `GPU`.
- `--pretrain_ckpt`: path of the `pretrain_checkpoint` file required for loading a weight of a pre-trained model parameter during incremental training or optimization.
- `--freeze_layer`: frozen network layer. Enter `none` or `backbone`.

### Running Shell Scripts

You can run the shell scripts `./scripts/run_train.sh` and `./scripts/run_eval.sh` on Linux. Input parameters on the interaction interface.

```bash
# Windows doesn't support Shell
# Linux train with Shell script
sh run_train.sh [PLATFORM] [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [RANK_TABLE_FILE] [DATASET_PATH] [CKPT_PATH](optional) [FREEZE_LAYER](optional) [FILTER_HEAD](optional)

# Linux eval with Shell script for fine tune
sh run_eval.sh [PLATFORM] [DATASET_PATH] [PRETRAIN_CKPT_PATH]
```

- `<PLATFORM>`: processor type. The default value is `Ascend`. You can set it to `GPU`.
- `<DEVICE_NUM>`: number of processes on each node (equivalent to a server or PC). You are advised to set this parameter to the number of Ascend AI Processors or GPUs on a server.
- `<VISIABLE_DEVICES(0,1,2,3,4,5,6,7)>`: device ID of character string type. During training, a process is bound to a device with the corresponding ID based on `<VISIABLE_DEVICES>`. Multiple device IDs are separated by commas (,). It is recommended that the number of IDs be the same as the number of processes.
- `<RANK_TABLE_FILE>`: a JSON file configured when platform is set to `Ascend`
- `<DATASET_PATH>`: path of the training or validation dataset. There is no default value. This parameter is mandatory for training or validation.
- `<CKPT_PATH>`: path of the checkpoint file required for loading a weight of a pre-trained model parameter during incremental training or optimization.
- `[FREEZE_LAYER]`: frozen network layer during fine-tuned model validation. Enter `none` or `backbone`.

## Loading Fine-Tuning Training

Only `train.py` can be run on Windows when MobileNetV2 is used for fine-tuning training. You can run the shell script `run_train.sh` and input [parameters](https://www.mindspore.cn/docs/programming_guide/en/r1.5/cv_mobilenetv2_fine_tune.html#id8) on Linux when MobileNetV2 is used for fine-tuning training.

The Windows system outputs information to an interactive command line. When running `run_train.sh` on the Linux system, use `&> <log_file_path>` at the end of the command line to write the standard output and error output to the log file. After the fine-tuning is successful, training starts. The training time and loss of each epoch are continuously written into the `./train/rank*/log*.log` file. If the fine-tuning fails, an error message is recorded in the preceding log file.

### Loading Training on CPU

- Set the number of nodes.

  Currently, `train.py` supports only a single processor. You do not need to adjust the number of processors. When the `run_train.sh` file is run, a single `CPU` is used by default. The number of CPUs cannot be changed.

- Start incremental training.

  Example 1: Use the python file to call a CPU.

    ```bash
    # Windows or Linux with Python
    python train.py --config_path ./default_config_cpu.yaml --platform CPU --dataset_path <TRAIN_DATASET_PATH>  --pretrain_ckpt ./pretrain_checkpoint/mobilenetv2_cpu_gpu.ckpt --freeze_layer backbone
    ```

  Example 2: Use the shell file to call a CPU.

    ```bash
    # Linux with Shell
    sh run_train.sh CPU [DATASET_PATH] [CKPT_PATH](optional) [FREEZE_LAYER](optional) [FILTER_HEAD](optional)
    ```

### Loading Training on GPU

- Set the number of nodes.

  Currently, `train.py` supports only a single processor. You do not need to adjust the number of nodes. When running the `run_train.sh` file, set `<nproc_per_node>` to the number of GPUs and `<visible_devices>` to IDs of available processors, that is, GPU IDs. You can select one or more device IDs and separate them with commas (,).

- Start incremental training.

    - Example 1: Use the python file to call a GPU.

        ```bash
        # Windows or Linux with Python
        python train.py --config_path ./default_config_gpu.yaml --platform GPU --dataset_path <TRAIN_DATASET_PATH> --pretrain_ckpt ./pretrain_checkpoint/mobilenetv2_cpu_gpu.ckpt --freeze_layer backbone
        ```

    - Example 2: Use the shell script to call a GPU whose device ID is `0`.

        ```bash
        # Linux with Shell
        sh run_train.sh GPU 1 0 [DATASET_PATH] [CKPT_PATH](optional) [FREEZE_LAYER](optional) [FILTER_HEAD](optional)
        ```

    - Example 3: Use the shell script to call eight GPUs whose device IDs are `0,1,2,3,4,5,6,7`.

        ```bash
        # Linux with Shell
        sh run_train.sh GPU 8 0,1,2,3,4,5,6,7 [DATASET_PATH] [CKPT_PATH](optional) [FREEZE_LAYER](optional) [FILTER_HEAD](optional)
        ```

### Loading Training on Ascend AI Processor

- Set the number of nodes.

  Currently, `train.py` supports only a single processor. You do not need to adjust the number of nodes. When running the `run_train.sh` file, set `<nproc_per_node>` to the number of Ascend AI Processors and `<visible_devices>` to IDs of available processors, that is, Ascend AI Processor IDs. You can select one or more device IDs from 0 to 7 on an 8-device server and separate them with commas (,). Currently, the number of Ascend AI Processors can only be set to 1 or 8.

- Start incremental training.

    - Example 1: Use the python file to call an Ascend AI Processor.

        ```bash
        # Windows or Linux with Python
        python train.py --config_path ./default_config.yaml --platform Ascend --dataset_path <TRAIN_DATASET_PATH>  --pretrain_ckpt  ./pretrain_checkpoint mobilenetv2_ascend.ckpt --freeze_layer backbone
        ```

    - Example 2: Use the shell script to call an Ascend AI Processor whose device ID is `0`.

        ```bash
        # Linux with Shell
        sh run_train.sh Ascend 1 0 ~/rank_table.json <TRAIN_DATASET_PATH> ../pretrain_checkpoint/mobilenetv2_ascend.ckpt backbone
        ```

    - Example 3: Use the shell script to call eight Ascend AI Processors whose device IDs are `0,1,2,3,4,5,6,7`.

        ```bash
        # Linux with Shell
        sh run_train.sh Ascend 8 0,1,2,3,4,5,6,7 ~/rank_table.json <TRAIN_DATASET_PATH> ../pretrain_checkpoint/mobilenetv2_ascend.ckpt backbone
        ```

### Fine-Tuning Training Result

- View the running result.

    - When running the python file, view the output information in the interactive command line. After running the shell script on `Linux`, run the `cat ./train/rank0/log0.log` command to view the output information. The output is as follows:

        ```text
        train args: Namespace(dataset_path='./dataset/train', platform='CPU', \
        pretrain_ckpt='./pretrain_checkpoint/mobilenetv2_cpu_gpu.ckpt', freeze_layer='backbone')
        cfg: {'num_classes': 26, 'image_height': 224, 'image_width': 224, 'batch_size': 150, \
        'epoch_size': 200, 'warmup_epochs': 0, 'lr_max': 0.03, 'lr_end': 0.03, 'momentum': 0.9, \
        'weight_decay': 4e-05, 'label_smooth': 0.1, 'loss_scale': 1024, 'save_checkpoint': True, \
        'save_checkpoint_epochs': 1, 'keep_checkpoint_max': 20, 'save_checkpoint_path': './', \
        'platform': 'CPU'}
        Processing batch: 16: 100%|███████████████████████████████████████████ █████████████████████| 16/16 [00:00<?, ?it/s]
        epoch[200], iter[16] cost: 256.030, per step time: 256.030, avg loss: 1.775total cos 7.2574 s
        ```

- Check the saved checkpoint files.

    - On Windows, run the `dir checkpoint` command to view the saved model files.

        ```text
        dir ckpt_0
        2020//0814 11:20        267,727 mobilenetv2_1.ckpt
        2020//0814 11:21        267,727 mobilenetv2_10.ckpt
        2020//0814 11:21        267,727 mobilenetv2_11.ckpt
        ...
        2020//0814 11:21        267,727 mobilenetv2_7.ckpt
        2020//0814 11:21        267,727 mobilenetv2_8.ckpt
        2020//0814 11:21        267,727 mobilenetv2_9.ckpt
        ```

    - On Linux, run the `ls ./checkpoint` command to view the saved model files.

        ```text
        ls ./ckpt_0/
        mobilenetv2_1.ckpt  mobilenetv2_2.ckpt
        mobilenetv2_3.ckpt  mobilenetv2_4.ckpt
        ...
        ```

## Validating the Fine-Tuning Training Model

### Validating the Model

Set mandatory [parameters](https://www.mindspore.cn/docs/programming_guide/en/r1.5/cv_mobilenetv2_fine_tune.html#id8) when using the validation set to test model performance. The default value of `--platform` is `Ascend`. You can set it to `CPU` or `GPU`. Finally, the standard output and error output are displayed in the interactive command line or written to the `eval.log` file.

```bash
# Windows/Linux with Python
python eval.py --config_path ./default_config_cpu.yaml --platform CPU --dataset_path <VAL_DATASET_PATH> --pretrain_ckpt ./ckpt_0/mobilenetv2_15.ckpt

# Linux with Shell
sh run_eval.sh CPU <VAL_DATASET_PATH> ../ckpt_0/mobilenetv2_15.ckpt
```

### Validation Result

When the python file is run, the validation result is output in the interactive command line. The shell script writes the information to `./eval.log`. You need to run the `cat ./eval.log` command to view the information. The result is as follows:

```text
result:{'acc': 0.9466666666666666666667}
pretrain_ckpt = ./ckpt_0/mobilenetv2_15.ckpt
```
