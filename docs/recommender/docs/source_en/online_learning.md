# Online Learning

<a href="https://gitee.com/mindspore/docs/blob/master/docs/recommender/docs/source_en/online_learning.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

The real-time update of the recommendation network model is one of the important technical indicators, and online learning can effectively improve the real-time update of the recommendation network model.

Key differences between online learning and offline training:

1. The dataset for online learning is streaming data with no definite dataset size, epoch, while the dataset for offline training has a definite data set size, epoch.
2. Online learning is in the form of a resident service, while the offline training exits tasks at the end of offline training.
3. Online learning requires collecting and storing training data, and driving the training process after a fixed amount of data has been collected or a fixed time window has elapsed.

## Overall Architecture

The user's streaming training data is pushed to kafka. MindPandas reads data from kafka and performs feature engineering transformation, and then writes to the feature storage engine. MindData reads data from the storage engine as training data for training. MindSpore, as a service resident, continuously receives data and performs training, with the overall process shown in the following figure:

![image.png](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/recommender/docs/source_en/images/online_training.png)

## Use Constraints

- Python 3.8 and above is required to be installed.
- Currently only GPU training, Linux operating system are supported.

## Python Package Dependencies

mindpandas  v0.1.0

mindspore_rec  v0.2.0

kafka-python v2.0.2

## Example

The following is an example of the process of online learning with the Criteo dataset training Wide&Deep. The sample code is located at [Online Learning](https://gitee.com/mindspore/recommender/tree/master/examples/online_learning).

MindSpore Recommender provides a specialized algorithm model `RecModel` for online learning, which is combined with MindPandas, a real-time data source Kafka for data reading and feature processing, to implement a simple online learning process.
First define a custom dataset for real-time data processing, where the constructor parameter `receiver` is of type `DataReceiver` in MindPands for receiving real-time data, and `__getitem__` means read data one at a time.

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

Then the above custom dataset is encapsulated into the online dataset required by `RecModel`.

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

After configuring the export strategy for the model Checkpoint, start the online training process.

```python
ckptconfig = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=5)
ckpoint_cb = ModelCheckpoint(prefix='widedeep_train', directory="./ckpt", config=ckptconfig)

model.online_train(dataset, callbacks=[TimeMonitor(1), callback, ckpoint_cb], dataset_sink_mode=True)
```

The following describes the start process for each module involved in the online learning process:

### Downloading Kafka

```bash
wget https://archive.apache.org/dist/kafka/3.2.0/kafka_2.13-3.2.0.tgz

tar -xzf kafka_2.13-3.2.0.tar.gz

cd kafka_2.13-3.2.0
```

To install other versions, please refer to <https://archive.apache.org/dist/kafka/>.

### Starting kafka-zookeeper

```bash
bin/zookeeper-server-start.sh config/zookeeper.properties
```

### Starting kafka-server

Open another command terminal and start the kafka service.

```bash
bin/kafka-server-start.sh config/server.properties
```

### Starting kafka_client

kafka_client needs to be started only once, and you can use kafka to set the number of partitions corresponding to the topic.

```bash
python kafka_client.py
```

### Start a Distributed Computing Engine

```bash
yrctl start --master  --address $MASTER_HOST_IP  

#Parameter description
--master： indicates that the current host is the master node. Non-master nodes do not need to specify the '--master' parameter
--address： ip of master node
```

### Starting Data producer

producer is used to simulate an online learning scenario where a local criteo dataset is written to kafka for use by the consumer. The current sample uses multiple processes to read two files and write the data to kafka.

```bash
python producer.py  --file1=$CRITEO_DATASET_FILE_PATH  --file2=$CRITEO_DATASET_FILE_PATH
#Parameter description
--file1： Path to the local disk for the criteo dataset
--file2： Path to the local disk for the criteo dataset
```

### Starting Data consumer

```bash
python consumer.py  --num_shards=$DEVICE_NUM  --address=$LOCAL_HOST_IP  --dataset_name=$DATASET_NAME
  --max_dict=$PATH_TO_VAL_MAX_DICT  --min_dict=$PATH_TO_CAT_TO_ID_DICT  --map_dict=$PATH_TO_VAL_MAP_DICT

#Parameter description
--num_shards： The number of device cards on the corresponding training side is set to 1 for single-card training and 8 for 8-card training.
--address： address of current sender
--dataset_name： dataset name
--namespace： channel name
--max_dict： Maximum dictionary of dense feature columns
--min_dict： Minimum dictionary of dense feature columns
--map_dict： Dictionary of sparse feature columns
```

The consumer needs 3 dataset-related files for feature engineering of criteo dataset: `all_val_max_dict.pkl`, `all_val_min_dict.pkl`, `cat2id_dict.pkl`, `$PATH_TO_VAL_MAX_DICT`, `$PATH _TO_CAT_TO_ID_DICT`, `$PATH_TO_VAL_MAP_DICT`, which are the absolute paths to these files on the environment, respectively. The specific production method of these 3 PKL files can be found in [process_data.py](https://gitee.com/mindspore/recommender/blob/master/datasets/criteo_1tb/process_data.py), switching the original criteo dataset to produce the corresponding .pkl files.

### Starting Online Training

For fhe yaml used by config, please refer to [default_config.yaml](https://gitee.com/mindspore/recommender/blob/master/examples/online_learning/default_config.yaml).

Single-card traininf:

```bash
python online_train.py --address=$LOCAL_HOST_IP   --dataset_name=criteo

#Parameter description:
--address： Local host ip. Receiving training data from MindPandas requires configuration
--dataset_name： Dataset name, consistent with the consumer module
```

Start with multi-card training MPI mode:

```bash
bash mpirun_dist_online_train.sh [$RANK_SIZE] [$LOCAL_HOST_IP]

#Parameter description:
RANK_SIZE：Number of multi-card training cards
LOCAL_HOST_IP：Local host ip for MindPandas to receive training data
```

Dynamic networking method to start multi-card training:

```bash
bash run_dist_online_train.sh [$WORKER_NUM] [$SHED_HOST] [$SCHED_PORT] [$LOCAL_HOST_IP]

#Parameter description:
WORKER_NUM：Number of multi-card training cards
SHED_HOST：IP of the Scheduler role required for MindSpore dynamic networking
SCHED_PORT：Port of the Scheduler role required for MindSpore dynamic networking
LOCAL_HOST_IP：Local host ip. Receiving training data from MindPandas requires configuration
```

When training is successfully started, the following log is output:

epoch and step represent the number of epoch and step corresponding to the current training step, and wide_loss and deep_loss represent the training loss values in the wide&deep network.

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
