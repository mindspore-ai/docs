# 在线学习

<a href="https://gitee.com/mindspore/docs/blob/master/docs/recommender/docs/source_zh_cn/online_learning.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

推荐网络模型更新的实时性是重要的技术指标之一，在线学习可有效提升推荐网络模型更新的实时性。

在线学习与离线训练的主要区别：

1. 在线学习的数据集为流式数据，无确定的dataset size、epoch，离线训练的数据集有确定的data set size、epoch。
2. 在线学习为常驻服务形式，离线训练结束后任务退出。
3. 在线学习需要收集并存储训练数据，收集到固定数量的数据或经过固定的时间窗口后驱动训练流程。

## 整体架构

用户的流式训练数据推送到kafka中，MindPandas从kafka读取数据并进行特征工程转换，然后写入特征存储引擎中，MindData从存储引擎中读取数据作为训练数据进行训练，MindSpore 作为服务常驻，持续接收数据并执行训练，整体流程如下图所示：

![image.png](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/recommender/docs/source_zh_cn/images/online_training.png)

## 使用约束

- 需要安装Python3.8及以上版本。
- 目前仅支持GPU训练、Linux操作系统。

## Python包依赖

mindpandas  v0.1.0

mindspore_rec  v0.2.0

kafka-python v2.0.2

## 使用样例

下面以Criteo数据集训练Wide&Deep为例，介绍一下在线学习的流程，样例代码位于[在线学习](https://gitee.com/mindspore/recommender/tree/master/examples/online_learning)。

MindSpore Recommender为在线学习提供了专门的算法模型`RecModel`，搭配实时数据源Kafka数据读取与特征处理的MindPandas即可实现一个简单的在线学习流程。
首先定义一个自定义的实时数据处理的数据集，其中的构造参数`receiver`是MindPands中的`DataReceiver`类型，用于接收实时数据，`__getitem__`表示一次读取一条数据。

```python
class StreamingDataset:
    def __init__(self, receiver):
      self.data_ = []
      self.receiver_ = receiver
      self.recv_data_cnt_ = 0

    def __getitem__(self, item):
      while not self.data_:
        data = self.receiver_.recv()
        self.recv_data_cnt_ += 1
        if data is not None:
          self.data_ = data.tolist()

      last_row = self.data_.pop()
      return np.array(last_row[0], dtype=np.int32), np.array(last_row[1], dtype=np.float32), np.array(last_row[2], dtype=np.float32)
```

接着将上述自定义数据集封装成`RecModel`所需要的在线数据集。

```python
from mindpandas.channel import DataReceiver
from mindspore_rec import RecModel as Model

receiver = DataReceiver(address=config.address, namespace=config.namespace,
                        dataset_name=config.dataset_name, shard_id=0)
stream_dataset = StreamingDataset(receiver)

dataset = ds.GeneratorDataset(stream_dataset, column_names=["id", "weight", "label"])
dataset = dataset.batch(config.batch_size)

train_net, _ = GetWideDeepNet(config)
train_net.set_train()

model = Model(train_net)
```

在配置好模型Checkpoint的导出策略后，启动在线训练进程。

```python
ckptconfig = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=5)
ckpoint_cb = ModelCheckpoint(prefix='widedeep_train', directory="./ckpt", config=ckptconfig)

model.online_train(dataset, callbacks=[TimeMonitor(1), callback, ckpoint_cb], dataset_sink_mode=True)
```

下面介绍在线学习流程中涉及各个模块的启动流程：

### 下载Kafka

```bash
wget https://archive.apache.org/dist/kafka/3.2.0/kafka_2.13-3.2.0.tgz

tar -xzf kafka_2.13-3.2.0.tgz

cd kafka_2.13-3.2.0
```

如需安装其他版本，请参照<https://archive.apache.org/dist/kafka/>。

### 启动kafka-zookeeper

```bash
bin/zookeeper-server-start.sh config/zookeeper.properties
```

### 启动kafka-server

打开另一个命令终端，启动kafka服务。

```bash
bin/kafka-server-start.sh config/server.properties
```

### 启动kafka_client

kafka_client只需要启动一次，可以使用kafka设置topic对应的partition数量。

```bash
python kafka_client.py
```

### 启动分布式计算引擎

```bash
yrctl start --master  --address $MASTER_HOST_IP  

#参数说明
--master： 表示当前host为master节点，非master节点不用指定‘--master’参数
--address： master节点的ip
```

### 启动数据producer

producer用于模拟在线学习场景，将本地的criteo数据集写入到kafka，供consumer使用。当前样例使用多进程读取两个文件，并将数据写入kafka。

```bash
python producer.py  --file1=$CRITEO_DATASET_FILE_PATH  --file2=$CRITEO_DATASET_FILE_PATH
#参数说明
--file1： criteo数据集在本地磁盘的存放路径
--file2： criteo数据集在本地磁盘的存放路径
```

### 启动数据consumer

```bash
python consumer.py  --num_shards=$DEVICE_NUM  --address=$LOCAL_HOST_IP  --dataset_name=$DATASET_NAME
  --max_dict=$PATH_TO_VAL_MAX_DICT  --min_dict=$PATH_TO_CAT_TO_ID_DICT  --map_dict=$PATH_TO_VAL_MAP_DICT

#参数说明
--num_shards： 对应训练侧的device 卡数，单卡训练则设置为1，8卡训练设置为8
--address： 当前sender的地址
--dataset_name： 数据集名称
--namespace： channel名称
--max_dict： 稠密特征列的最大值字典
--min_dict： 稠密特征列的最小值字典
--map_dict： 稀疏特征列的字典
```

consumer为criteo数据集进行特征工程需要3个数据集相关文件：`all_val_max_dict.pkl`、`all_val_min_dict.pkl`、`cat2id_dict.pkl`、`$PATH_TO_VAL_MAX_DICT`、`$PATH_TO_CAT_TO_ID_DICT`、`$PATH_TO_VAL_MAP_DICT` 分别为这些文件在环境上的绝对路径。这3个pkl文件具体生产方法可以参考[process_data.py](https://gitee.com/mindspore/recommender/blob/master/datasets/criteo_1tb/process_data.py)，对原始criteo数据集做转换生产对应的.pkl文件。

### 启动在线训练

config采用yaml的形式，见[default_config.yaml](https://gitee.com/mindspore/recommender/blob/master/examples/online_learning/default_config.yaml)。

单卡训练：

```bash
python online_train.py --address=$LOCAL_HOST_IP   --dataset_name=criteo

#参数说明：
--address： 本机host ip，从MindPandas接收训练数据需要配置
--dataset_name： 数据集名字，和consumer模块保持一致
```

多卡训练MPI方式启动：

```bash
bash mpirun_dist_online_train.sh [$RANK_SIZE] [$LOCAL_HOST_IP]

#参数说明：
RANK_SIZE：多卡训练卡数量
LOCAL_HOST_IP：本机host ip，用于MindPandas接收训练数据
```

动态组网方式启动多卡训练：

```bash
bash run_dist_online_train.sh [$WORKER_NUM] [$SHED_HOST] [$SCHED_PORT] [$LOCAL_HOST_IP]

#参数说明：
WORKER_NUM：多卡训练卡数量
SHED_HOST：MindSpore动态组网需要的Scheduler 角色的IP
SCHED_PORT：MindSpore动态组网需要的Scheduler 角色的Port
LOCAL_HOST_IP：本机host ip，从MindPandas接收训练数据需要配置
```

成功启动训练后，会输出如下日志：

其中epoch和step表示当前训练步骤对应的epoch和step数，wide_loss和deep_loss表示wide&deep网络中的训练loss值。

```text
epoch: 1, step: 1, wide_loss: 0.66100323, deep_loss: 0.72502613
epoch: 1, step: 2, wide_loss: 0.46781272, deep_loss: 0.5293098
epoch: 1, step: 3, wide_loss: 0.363207, deep_loss: 0.42204413
epoch: 1, step: 4, wide_loss: 0.3051032, deep_loss: 0.36126155
epoch: 1, step: 5, wide_loss: 0.24045062, deep_loss: 0.29395688
epoch: 1, step: 6, wide_loss: 0.24296054, deep_loss: 0.29386574
epoch: 1, step: 7, wide_loss: 0.20943595, deep_loss: 0.25780612
epoch: 1, step: 8, wide_loss: 0.19562452, deep_loss: 0.24153553
epoch: 1, step: 9, wide_loss: 0.16500896, deep_loss: 0.20854339
epoch: 1, step: 10, wide_loss: 0.2188702, deep_loss: 0.26011512
epoch: 1, step: 11, wide_loss: 0.14963374, deep_loss: 0.18867904
```
