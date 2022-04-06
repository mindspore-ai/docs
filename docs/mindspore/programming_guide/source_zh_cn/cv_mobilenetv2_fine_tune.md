# 使用MobileNetV2网络实现微调（Fine Tune）

`Ascend` `GPU` `CPU` `全流程`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/cv_mobilenetv2_fine_tune.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

## 概述

计算机视觉任务中，从头开始训练一个网络耗时巨大，需要大量计算能力。预训练模型选择的常见的OpenImage、ImageNet、VOC、COCO等公开大型数据集，规模达到几十万甚至超过上百万张。大部分任务数据规模较大，训练网络模型时，如果不使用预训练模型，从头开始训练网络，需要消耗大量的时间与计算能力，模型容易陷入局部极小值和过拟合。因此大部分任务都会选择预训练模型，在其上做微调（也称为Fine Tune）。

MindSpore是一个多元化的机器学习框架。既可以在手机等端侧和PC等设备上运行，也可以在云上的服务器集群上运行。目前MobileNetV2支持在Windows、EulerOS和Ubuntu系统中使用单个CPU做微调，也可以使用单个或者多个Ascend AI处理器或GPU做微调，本教程将会介绍如何在不同系统与处理器下的MindSpore框架中做微调的训练与验证。

目前，Window上暂只支持支持CPU，Ubuntu与EulerOS上支持CPU、GPU与Ascend AI处理器三种处理器。

> 你可以在这里找到完整可运行的样例代码：<https://gitee.com/mindspore/models/tree/master/official/cv/mobilenetv2>

## 任务描述及准备

### 环境配置

若在本地环境运行，需要安装MindSpore框架，配置CPU、GPU或Ascend AI处理器。若在华为云环境上运行，不需要安装MindSpore框架，不需要配置Ascend AI处理器、CPU与GPU，可以跳过本小节。

Windows操作系统中使用`\`，Linux操作系统中使用`/`分割路径地址中不同层级目录，下文中默认使用`/`，若用户使用Windows操作系统，路径地址中`/`需自行更改为`\`。

1. 安装MindSpore框架
    在EulerOS、Ubuntu或者Windows等系统上需要根据系统和处理器架构[安装对应版本MindSpore框架](https://www.mindspore.cn/install)。

2. 配置CPU环境  
    使用CPU时，在代码中，需要在调用CPU开始训练或测试前，按照如下代码设置：

    ```python
    if config.platform == "CPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, \
            save_graphs=False)
    ```

3. 配置GPU环境  
    使用GPU时，在代码中，需要在调用GPU开始训练或测试前，按照如下代码设置：

    ```python
    elif config.platform == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, save_graphs=False)
        if config.run_distribute:
            init("nccl")
            context.set_auto_parallel_context(device_num=get_group_size(),
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    ```

4. 配置Ascend环境  
    以Ascend 910 AI处理器为例，1个8个处理器环境的json配置文件`hccl_config.json`示例如下。单/多处理器环境可以根据以下示例调整`"server_count"`与`device`：

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

    使用Ascend AI处理器时，在代码中，需要在调用Ascend AI处理器开始训练或测试前，按照如下代码设置：

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

### 下载代码

在Gitee中克隆[Models开源项目仓库](https://gitee.com/mindspore/models.git)，进入`./models/official/cv/mobilenetv2/`。

```bash
git clone https://gitee.com/mindspore/models.git
cd ./models/official/cv/mobilenetv2
```

代码结构如下：

```text
├── MobileNetV2
  ├── README.md                  # MobileNetV2相关描述
  ├── ascend310_infer            # 用于310推理
  ├── scripts
  │   ├──run_train.sh            # 使用CPU、GPU或Ascend进行训练、微调或增量学习的shell脚本
  │   ├──run_eval.sh             # 使用CPU、GPU或Ascend进行评估的shell脚本
  │   ├──cache_util.sh           # 包含一些使用cache的帮助函数
  │   ├──run_train_nfs_cache.sh  # 使用NFS的数据集进行训练并利用缓存服务进行加速的shell脚本
  │   ├──run_infer_310.sh        # 使用Dvpp 或CPU算子进行推理的shell脚本
  ├── src
  │   ├──aipp.cfg                # aipp配置
  │   ├──dataset.py              # 创建数据集
  │   ├──launch.py               # 启动Python脚本
  │   ├──lr_generator.py         # 配置学习率
  │   ├──mobilenetV2.py          # MobileNetV2架构
  │   ├──models.py               # 加载define_net、Loss、及Monitor
  │   ├──utils.py                # 加载ckpt_file进行微调或增量学习
  │   └──model_utils
  │      ├──config.py            # 获取.yaml配置参数
  │      ├──device_adapter.py    # 获取云上id
  │      ├──local_adapter.py     # 获取本地id
  │      └──moxing_adapter.py    # 云上数据准备
  ├── default_config.yaml        # 训练配置参数(ascend)
  ├── default_config_cpu.yaml    # 训练配置参数(cpu)
  ├── default_config_gpu.yaml    # 训练配置参数(gpu)
  ├── train.py                   # 训练脚本
  ├── eval.py                    # 评估脚本
  ├── export.py                  # 模型导出脚本
  ├── mindspore_hub_conf.py      # MindSpore Hub接口
  ├── postprocess.py             # 推理后处理脚本
```

运行微调训练与测试时，Windows、Ubuntu与EulersOS上可以使用Python文件`train.py`与`eval.py`，Ubuntu与EulerOS上还可以使用Shell脚本文件`run_train.sh`与`run_eval.sh`。

使用脚本文件`run_train.sh`时，该文件会将运行`launch.py`并且将参数传入`launch.py`，`launch.py`根据分配的CPU、GPU或Ascend AI处理器数量，启动单个/多个进程运行`train.py`，每一个进程分配对应的一个处理器。

### 准备预训练模型

用户需要根据不同处理器种类[下载CPU/GPU预训练模型](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2_cpu_gpu.ckpt)或[下载Ascend预训练模型](https://download.mindspore.cn/model_zoo/r1.2/mobilenetv2_ascend_v120_imagenet2012_official_cv_bs256_acc71/mobilenetv2_ascend_v120_imagenet2012_official_cv_bs256_acc71.ckpt)到以下目录：  
`./pretrain_checkpoint/`

- CPU/GPU 处理器

    ```bash
    mkdir pretrain_checkpoint
    wget -P ./pretrain_checkpoint https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2_cpu_gpu.ckpt --no-check-certificate
    ```

- Ascend AI处理器

    ```bash
    mkdir pretrain_checkpoint
    wget -P ./pretrain_checkpoint https://download.mindspore.cn/model_zoo/r1.2/mobilenetv2_ascend_v120_imagenet2012_official_cv_bs256_acc71/mobilenetv2_ascend_v120_imagenet2012_official_cv_bs256_acc71.ckpt --no-check-certificate
    ```

### 准备数据

准备ImageFolder格式管理的数据集，运行`run_train.sh`时加入`<dataset_path>`参数，运行`train.py`时加入`--dataset_path <dataset_path>`参数：

数据集结构如下：

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

> 运行本例时，上述结构中验证数据集文件夹名称`eval`需改为`validation_preprocess`。

## 预训练模型加载代码详解

在微调时，需要加载预训练模型。不同数据集和任务中特征提取层（卷积层）分布趋于一致，但是特征向量的组合（全连接层）不相同，分类数量（全连接层output_size）通常也不一致。在微调时，只加载与训练特征提取层参数，不加载与训练全连接层参数；在微调与初始训练时，加载与训练特征提取层参数与全连接层参数。

在训练与测试之前，首先按照代码第1行，构建MobileNetV2的backbone网络，head网络，并且构建包含这两个子网络的MobileNetV2网络。代码第3-10行展示了如何定义`backbone_net`与`head_net`，以及将两个子网络置入`mobilenet_v2`中。代码第12-27行，展示了在微调训练模式下，需要将预训练模型加载入`backbone_net`子网络，并且冻结`backbone_net`中的参数，不参与训练。代码第25-27行展示了如何冻结网络参数。

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

## 参数简介

每个参数需要用户根据自己本地的处理器类型、数据地址与预训练模型地址等修改为相应的值。

### 运行Python文件

在Windows与Linux系统上训练时，运行`train.py`时需要传入 `config_path`、 `dataset_path`、`platform`、`pretrain_ckpt`与`freeze_layer`五个参数。验证时，运行`eval.py`并且传入`config_path`、`dataset_path`、`platform`、`pretrain_ckpt`四个参数。

```bash
# Windows/Linux train with Python file
python train.py --config_path [CONFIG_PATH] --platform [PLATFORM] --dataset_path <DATASET_PATH>  --pretrain_ckpt [PRETRAIN_CHECKPOINT_PATH] --freeze_layer[("none", "backbone")]

# Windows/Linux eval with Python file
python eval.py --config_path [CONFIG_PATH] --platform [PLATFORM] --dataset_path <DATASET_PATH> --pretrain_ckpt <PRETRAIN_CHECKPOINT_PATH>
```

- `--config_path`：训练与验证所需参数。
- `--dataset_path`：训练与验证数据集地址，无默认值，用户训练/验证时必须输入。
- `--platform`：处理器类型，默认为“Ascend”，可以设置为“CPU”或"GPU"。
- `--pretrain_ckpt`：增量训练或调优时，需要传入pretrain_checkpoint文件路径以加载预训练好的模型参数权重。
- `--freeze_layer`：冻结网络层，输入“none"、"backbone"其中一个。

### 运行Shell脚本

在Linux系统上时，可以选择运行Shell脚本文件`./scripts/run_train.sh`与`./scripts/run_eval.sh`。运行时需要在交互界面中同时传入参数。

```bash
# Windows doesn't support Shell
# Linux train with Shell script
sh run_train.sh [PLATFORM] [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [RANK_TABLE_FILE] [DATASET_PATH] [CKPT_PATH](optional) [FREEZE_LAYER](optional) [FILTER_HEAD](optional)

# Linux eval with Shell script for fine tune
sh run_eval.sh [PLATFORM] [DATASET_PATH] [PRETRAIN_CKPT_PATH]
```

- `<PLATFORM>`：处理器类型，默认为“Ascend”，可以设置为“GPU”。
- `<DEVICE_NUM>`：每个节点（一台服务器/PC相当于一个节点）进程数量，建议设置为机器上Ascend AI处理器数量或GPU数量。
- `<VISIABLE_DEVICES(0,1,2,3,4,5,6,7)>`：字符串格式的设备ID，训练将会根据`<VISIABLE_DEVICES>`将进程绑定到对应ID的设备上，多个设备ID之间使用','分隔，建议ID数量与进程数量相同。
- `<RANK_TABLE_FILE>`：platform选择Ascend时，需要配置Ascend的配置Json文件,。
- `<DATASET_PATH>`：训练与验证数据集地址，无默认值，用户训练/验证时必须输入。
- `<CKPT_PATH>`：增量训练或调优时，需要传入checkpoint文件路径以加载预训练好的模型参数权重
- `[FREEZE_LAYER]`：针对微调的模型做验证时，需要选择不冻结网络或者冻结backbone。

## 加载微调训练

Windows系统上，MobileNetV2做微调训练时，只能运行`train.py`。Linux系统上，使用MobileNetV2做微调训练时，可以选择运行`run_train.sh`， 并在运行Shell脚本文件时传入[参数](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/cv_mobilenetv2_fine_tune.html#参数简介)。

Windows系统输出信息到交互式命令行，Linux系统环境下运行`run_train.sh`时，命令行结尾使用`&> <log_file_path>`将标准输出与错误输出写入log文件。微调成功开始训练，`./train/rank*/log*.log`中会持续写入每一个epoch的训练时间与Loss等信息。若未成功，上述log文件会写入失败报错信息。

### CPU加载训练

- 设置节点数量

  目前运行`train.py`时仅支持单处理器，不需要调整处理器数量。运行`run_train.sh`文件时，`CPU`设备默认为单处理器，目前暂不支持修改CPU数量。

- 开始增量训练

  使用样例1：通过Python文件调用1个CPU处理器。

    ```bash
    # Windows or Linux with Python
    python train.py --config_path ./default_config_cpu.yaml --platform CPU --dataset_path <TRAIN_DATASET_PATH> --pretrain_ckpt ./pretrain_checkpoint/mobilenetv2_cpu_gpu.ckpt --freeze_layer backbone --filter_head FILTER_HEAD &> ./train.log &
    ```

  使用样例2：通过Shell文件调用1个CPU处理器。

    ```bash
    # Linux with Shell
    sh run_train.sh CPU [DATASET_PATH] [CKPT_PATH](optional) [FREEZE_LAYER](optional) [FILTER_HEAD](optional)
    ```

### GPU加载训练

- 设置节点数量

  目前运行`train.py`时仅支持单处理器，不需要调整节点数量。运行`run_train.sh`文件时，设置`<nproc_per_node>`为GPU数量， `<visible_devices>`为可使用的处理器编号，即GPU的ID，可以选择一个或多个设备ID，使用`,`隔开。

- 开始增量训练

    - 使用样例1：通过Python文件调用1个GPU处理器。

        ```bash
        # Windows or Linux with Python
        python train.py  --config_path ./default_config_gpu.yaml --platform GPU --dataset_path <TRAIN_DATASET_PATH> --pretrain_ckpt ./pretrain_checkpoint/mobilenetv2_cpu_gpu.ckpt --freeze_layer backbone
        ```

    - 使用样例2：通过Shell脚本调用1个GPU处理器，设备ID为`“0”`。

        ```bash
        # Linux with Shell
        sh run_train.sh GPU 1 0 [DATASET_PATH] [CKPT_PATH](optional) [FREEZE_LAYER](optional) [FILTER_HEAD](optional)
        ```

    - 使用样例3：通过Shell脚本调用8个GPU处理器，设备ID为`“0,1,2,3,4,5,6,7”`。

        ```bash
        # Linux with Shell
        sh run_train.sh GPU 8 0,1,2,3,4,5,6,7 [DATASET_PATH] [CKPT_PATH](optional) [FREEZE_LAYER](optional) [FILTER_HEAD](optional)
        ```

### Ascend加载训练

- 设置节点数量

  目前运行`train.py`时仅支持单处理器，不需要调整节点数量。运行`run_train.sh`文件时，设置`<nproc_per_node>`为Ascend AI处理器数量， `<visible_devices>`为可使用的处理器编号，即Ascend AI处理器的ID，8卡服务器可以选择0-7中一个或多个设备ID，使用`,`隔开。Ascend节点处理器数量目前只能设置为1或者8。

- 开始增量训练

    - 使用样例1：通过Python文件调用1个Ascend处理器。

        ```bash
        # Windows or Linux with Python
        python train.py --config_path ./default_config.yaml --platform Ascend --dataset_path <TRAIN_DATASET_PATH>  --pretrain_ckpt  ./pretrain_checkpoint/mobilenetv2_ascend_v120_imagenet2012_official_cv_bs256_acc71.ckpt --freeze_layer backbone
        ```

    - 使用样例2：通过Shell脚本调用1个Ascend AI处理器，设备ID为“0”。

        ```bash
        # Linux with Shell
        sh run_train.sh Ascend 1 0 ~/rank_table.json <TRAIN_DATASET_PATH> ../pretrain_checkpoint/mobilenetv2_ascend_v120_imagenet2012_official_cv_bs256_acc71.ckpt backbone
        ```

    - 使用样例3：通过Shell脚本调用8个Ascend AI处理器，设备ID为”0,1,2,3,4,5,6,7“。

        ```bash
        # Linux with Shell
        sh run_train.sh Ascend 8 0,1,2,3,4,5,6,7 ~/rank_table.json <TRAIN_DATASET_PATH> ../pretrain_checkpoint/mobilenetv2_ascend_v120_imagenet2012_official_cv_bs256_acc71.ckpt backbone
        ```

### 微调训练结果

- 查看运行结果。

    - 运行Python文件时在交互式命令行中查看打印信息，`Linux`上运行Shell脚本运行后使用`cat ./train/rank0/log0.log`中查看打印信息，输出结果如下：

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

- 查看保存的checkpoint文件。

    - Windows上使用`dir checkpoint`查看保存的模型文件：

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

    - Linux上使用`ls ./checkpoint`查看保存的模型文件：

        ```text
        ls ./ckpt_0/
        mobilenetv2_1.ckpt  mobilenetv2_2.ckpt
        mobilenetv2_3.ckpt  mobilenetv2_4.ckpt
        ...
        ```

## 验证微调训练模型

### 验证模型

使用验证集测试模型性能，需要输入必要[参数](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/cv_mobilenetv2_fine_tune.html#参数简介)，`--platform`默认为“Ascend”，可自行设置为"CPU"或"GPU"。最终在交互式命令行中展示标准输出与错误输出，或者将其写入`eval.log`文件。

```bash
# Windows/Linux with Python
python eval.py --config_path ./default_config_cpu.yaml --platform CPU --dataset_path <VAL_DATASET_PATH> --pretrain_ckpt ./ckpt_0/mobilenetv2_15.ckpt

# Linux with Shell
sh run_eval.sh CPU <VAL_DATASET_PATH> ../ckpt_0/mobilenetv2_15.ckpt
```

### 验证结果

运行Python文件时在交互式命令行中输出验证结果，Shell脚本将把这些信息写入`./eval.log`中，需要使用`cat ./eval.log`查看，结果如下：

```text
result:{'acc': 0.9466666666666666666667}
pretrain_ckpt = ./ckpt_0/mobilenetv2_15.ckpt
```
