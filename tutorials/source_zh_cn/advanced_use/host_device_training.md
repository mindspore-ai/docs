# Host+Device混合训练

<!-- TOC -->

- [Host+Device混合训练](#Host+Device混合训练)
    - [概述](#概述)
    - [准备工作](#准备工作)
    - [配置混合执行](#配置混合执行)
    - [训练模型](#训练模型)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/advanced_use/host_device_training.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

在深度学习中，工作人员时常会遇到超大模型的训练问题，即模型参数所占内存超过了设备内存上限。为高效地训练超大模型，一种方案便是[分布式并行训练](https://www.mindspore.cn/tutorial/zh-CN/master/advanced_use/distributed_training.html)，也就是将工作交由同构的多个加速器（如Ascend 910 AI处理器，GPU等）共同完成。但是这种方式在面对几百GB甚至几TB级别的模型时，所需的加速器过多。而当从业者实际难以获取大规模集群时，这种方式难以应用。另一种可行的方案是使用主机端（Host）和加速器（Device）的混合训练模式。此方案同时发挥了主机端内存大和加速器端计算快的优势，是一种解决超大模型训练较有效的方式。

在MindSpore中，用户可以将待训练的参数放在主机，同时将必要算子的执行位置配置为主机，其余算子的执行位置配置为加速器，从而方便地实现混合训练。此教程以推荐模型[Wide&Deep](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/wide_and_deep)为例，讲解MindSpore在主机和Ascend 910 AI处理器的混合训练。

## 准备工作

1. 准备模型代码。Wide&Deep的代码可参见：<https://gitee.com/mindspore/mindspore/tree/master/model_zoo/wide_and_deep>，其中，`train_and_eval_auto_parallel.py`为训练的主函数所在，`src/`目录中包含Wide&Deep模型的定义、数据处理和配置信息等，`script/`目录中包含不同配置下的训练脚本。

2. 准备数据集。数据集下载链接：<https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz>。利用脚本`/src/preprocess_data.py`将数据集转换为MindRecord格式。

3. 配置处理器信息。在裸机环境（即本地有Ascend 910 AI 处理器）进行分布式训练时，需要配置加速器信息文件。此样例只使用一个加速器，故只需配置包含0号卡的`rank_table_1p_0.json`文件（每台机器的具体的IP信息不同，需要查看网络配置来设定，此为示例），如下所示：

    ```json
    {
         "board_id": "0x0020",
         "chip_info": "910",
         "deploy_mode": "lab",
         "group_count": "1",
         "group_list": [
             {
                 "device_num": "1",
                 "server_num": "1",
                 "group_name": "",
                 "instance_count": "1",
                 "instance_list": [
                          {"devices":[{"device_id":"0","device_ip":"192.1.113.246"}],"rank_id":"0","server_id":"10.155.170.16"}
                    ]
             }
         ],
         "para_plane_nic_location": "device",
         "para_plane_nic_name": [
             "eth0"
         ],
         "para_plane_nic_num": "1",
         "status": "completed"
     }
    
    ```

## 配置混合执行

1. 配置待训练参数的存储位置。在`train_and_eval_auto_parallel.py`文件`train_and_eval`函数的`model.train`调用中，增加配置`dataset_sink_mode=False`，以指示参数数据保持在主机端，而非加速器端。

2. 配置待训练参数的稀疏性质。由于待训练参数的规模大，需将参数配置为稀疏，也就是：真正参与计算的并非全量的参数，而是其索引值。由于稀疏特性在MindSpore中属于正在开发中的特性，其用法可能会有优化，此文档也会更新。
    
    在`src/wide_and_deep.py`的`class WideDeepModel(nn.Cell)`中，将`self.wide_w`赋值替换为
    
    ```python
    self.wide_w = Parameter(Tensor(np.random.normal(loc=0.0, scale=0.01, size=[184968, 1]).astype(dtype=np_type)), name='Wide_w', sparse_grad='Wide_w')
    ``` 
    
    将`self.embedding_table`赋值替换为
    
    ```python
    self.embedding_table = Parameter(Tensor(np.random.normal(loc=0.0, scale=0.01, size=[184968, 80]).astype(dtype=np_type)), name='V_l2',  sparse_grad='V_l2')
    ```
    
    在`src/wide_and_deep.py`文件的`class WideDeepModel(nn.Cell)`类的`construct`函数中，将函数的返回值替换为如下值，以适配参数的稀疏性：
    
    ```
    return out, deep_id_embs
    ```
    
    除此之外，还需要配置对应的环境变量：
    
    ```shell
    export UNDETERMINED_SPARSE_SHAPE_TYPES="Wide_w:624000:Int32:624000 1:Float32:184968 1;V_l2:624000:Int32:624000 80:Float32:184968 80"
    ```

3. 配置必要算子和优化器的执行位置。在`src/wide_and_deep.py`的`class WideDeepModel(nn.Cell)`中，为`GatherV2`增加配置主机端执行的属性，

    ```python
    self.gather_v2 = P.GatherV2().add_prim_attr('primitive_target', 'CPU')
    ```
    
    在`src/wide_and_deep.py`文件的`class TrainStepWrap(nn.Cell)`中，为两个优化器增加配置主机端执行的属性。
    
    ```python
    self.optimizer_w.sparse_opt.add_prim_attr('primitive_target', 'CPU')
    ```
    
    ```python
    self.optimizer_d.sparse_opt.add_prim_attr('primitive_target', 'CPU')
    ```

## 训练模型

使用训练脚本`script/run_auto_parallel_train.sh`，
执行命令：`bash run_auto_parallel_train.sh 1 1 DATASET RANK_TABLE_FILE MINDSPORE_HCCL_CONFIG_PATH`，
其中第一个`1`表示用例使用的卡数，第二`1`表示训练的epoch数，`DATASET`是数据集所在路径，`RANK_TABLE_FILE`和`MINDSPORE_HCCL_CONFIG_PATH`为上述`rank_table_1p_0.json`文件所在路径。

运行日志保存在`device_0`目录下，其中`loss.log`保存一个epoch内中多个loss值，如下：

```
epoch: 1 step: 1, wide_loss is 0.6873926, deep_loss is 0.8878349
epoch: 1 step: 2, wide_loss is 0.6442529, deep_loss is 0.8342661
epoch: 1 step: 3, wide_loss is 0.6227323, deep_loss is 0.80273706
epoch: 1 step: 4, wide_loss is 0.6107221, deep_loss is 0.7813441
epoch: 1 step: 5, wide_loss is 0.5937832, deep_loss is 0.75526017
epoch: 1 step: 6, wide_loss is 0.5875453, deep_loss is 0.74038756
epoch: 1 step: 7, wide_loss is 0.5798845, deep_loss is 0.7245408
epoch: 1 step: 8, wide_loss is 0.57553077, deep_loss is 0.7123517
epoch: 1 step: 9, wide_loss is 0.5733629, deep_loss is 0.70278376
epoch: 1 step: 10, wide_loss is 0.566089, deep_loss is 0.6884129
```

`test_deep0.log`保存pytest进程输出的详细的运行时日志（需要将日志级别设置为INFO，且在MindSpore编译时加上-p on选项），搜索关键字`EmbeddingLookup`，可找到如下信息：

```
[INFO] DEVICE(109904,python3.7):2020-06-27-12:42:34.928.275 [mindspore/ccsrc/device/cpu/cpu_kernel_runtime.cc:324] Run] Call Default/network-VirtualDatasetCellTriple/_backbone-NetWithLossClass/network-WideDeepModel/EmbeddingLookup-op297 in 3066 us.
[INFO] DEVICE(109904,python3.7):2020-06-27-12:42:34.943.896 [mindspore/ccsrc/device/cpu/cpu_kernel_runtime.cc:324] Run] Call Default/network-VirtualDatasetCellTriple/_backbone-NetWithLossClass/network-WideDeepModel/EmbeddingLookup-op298 in 15521 us.
```

表示`EmbeddingLookup`在主机端的执行时间。由于`GatherV2`算子在主机端执行时会被换成`EmbeddingLookup`算子，这也是`GatherV2`的执行时间。
继续在`test_deep0.log`搜索关键字`SparseApplyFtrl`和`SparseApplyLazyAdam`，可找到如下信息：

```
[INFO] DEVICE(109904,python3.7):2020-06-27-12:42:35.422.963 [mindspore/ccsrc/device/cpu/cpu_kernel_runtime.cc:324] Run] Call Default/optimizer_w-FTRL/SparseApplyFtrl-op299 in 54492 us.
[INFO] DEVICE(109904,python3.7):2020-06-27-12:42:35.565.953 [mindspore/ccsrc/device/cpu/cpu_kernel_runtime.cc:324] Run] Call Default/optimizer_d-LazyAdam/SparseApplyLazyAdam-op300 in 142865 us.
```

表示两个优化器在主机端的执行时间。