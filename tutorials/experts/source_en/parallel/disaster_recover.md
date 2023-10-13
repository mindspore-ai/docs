# Disaster Recovery in Dynamic Cluster Scenarios

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/tutorials/experts/source_en/parallel/disaster_recover.md)

## Overview

The model training has high reliability and serviceability requirements for distributed training architecture. MindSpore dynamic cluster startup method supports data parallel disaster recovery. There is process abnormal exit in multi-card data parallel training scenarios cluster (multiple Workers and a Scheduler), after the process is pulled up again, the training task continues to be able to execute normally.

Specifically, in the graph mode, the data sink mode is used for training, and the data parallel mode is turned on. After the training cluster is started by dynamic cluster, if any process exits abnormally during the training process, it is guaranteed that the training can be continued after pulling up the corresponding script of the corresponding process under the same environment variables (`MS_ENABLE_RECOVERY` and `MS_RECOVERY_PATH`) and the accuracy convergence will not be affected.

> Disaster recovery in dynamic cluster scenarios only supports GPUs and needs to run in Graph mode.

For more detailed instructions, see dynamic cluster environment variables in the [Environment Variables List](https://www.mindspore.cn/docs/en/r2.2/note/env_var_list.html).

## Operation Practice

The following is an example of how to do this with Ascend:

### Sample Code Description

> Download the full sample code: [disaster_recover](https://gitee.com/mindspore/docs/tree/r2.2/docs/sample_code/disaster_recover).

The directory structure is as follows:

```text
└─ sample_code
    ├─ disaster_recover
       ├── train.py
       ├── run.sh
       └── recover.sh
    ...
```

`train.py` is the script that defines the network structure and the training process. `run.sh` is the execution script and `recover.sh` is the recovery script after node failure.

### Network Structure

The network structure and dataset loading is consistent with the example in [Dynamic Cluster Startup Method](https://www.mindspore.cn/tutorials/experts/en/r2.2/parallel/dynamic_cluster.html).

### Defining the Training Process

```python
import mindspore as ms
from mindspore import nn, train

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()
loss_cb = train.LossMonitor(20)
# Configure the interval at which checkpoints are saved and the maximum number to be saved
ckpt_config = train.CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=5)
# Configure the checkpoint storage path, and each process uses different paths
ckpoint_cb = train.ModelCheckpoint(prefix='train', directory="./ckpt_of_rank/"+str(get_rank()), config=ckpt_config)
model = ms.Model(net, loss_fn=loss_fn, optimizer=optimizer)
model.train(10, data_set, callbacks=[loss_cb, ckpoint_cb])
```

Each worker is enabled to save checkpoints with different paths (e.g., the directory setting in the above example uses a rank id to ensure that the paths are not the same) to prevent checkpoint saving conflicts with the same name. Checkpoint is used for abnormal process recovery and normal process rollback, and the rollback of training means that each Worker in the cluster is restored to the state corresponding to the latest checkpoint, and at the same time, the data side is also rolled back to the corresponding step, and then continue to train.

The interval between checkpoints is configurable, and determines the granularity of disaster recovery. The smaller the interval, the smaller the number of steps back to the last checkpoint save, but frequent checkpoint saves may also affect the training efficiency, and larger intervals have the opposite effect. keep_checkpoint_max is set to at least 2 (to prevent checkpoint saves from failing).

### Preparing the Startup Script

The script content `run.sh` is as follows, adding environment variables related to disaster recovery:

```bash
EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

export MS_ENABLE_RECOVERY=1                # Enable disaster recovery
export MS_RECOVERY_PATH=./recovery/        # Set the disaster recovery file save path

rm -rf device
mkdir device
echo "start training"

# Loop start of 8 Worker training processes
for((i=0;i<8;i++));
do
    export MS_WORKER_NUM=8          # Set the number of Worker processes in the cluster to 8
    export MS_SCHED_HOST=127.0.0.1  # Set the Scheduler IP address to the local loop address
    export MS_SCHED_PORT=8118       # Set the Scheduler Port
    export MS_ROLE=MS_WORKER        # Set the startup process to MS_WORKER role
    export MS_NODE_ID=$i            # Set the process id, optional
    python ./train.py > device/worker_$i.log 2>&1 &     # Start training scripts
done

# Start 1 Scheduler process
export MS_WORKER_NUM=8              # Set the number of Worker processes in the cluster to 8
export MS_SCHED_HOST=127.0.0.1      # Set the Scheduler IP address to the local loop address
export MS_SCHED_PORT=8118           # Set the Scheduler Port
export MS_ROLE=MS_SCHED             # Set the startup process to MS_SCHED role
python ./train.py > device/scheduler.log 2>&1 &     # Start training scripts
```

The environment variable `MS_ENABLE_RECOVERY=1` indicates that disaster recovery is enabled, and `MS_RECOVERY_PATH=. /recovery/` means configuring the path where persistence files are stored.

Before starting the Worker and Scheduler, you need to add the relevant environment variable settings, for example:

- `MS_WORKER_NUM=8`: Configure the number of Worker processes to 8.
- `MS_SCHED_HOST=127.0.0.1`: Configure the Scheduler IP address to 127.0.0.1.
- `MS_SCHED_PORT=8118`: Configure the port number of the Scheduler to be 8118.
- `MS_ROLE=MS_WORKER`: Configure the role of the current process. `MS_WORKER` means the role is Worker, and `MS_SCHED` means the role is Scheduler.

Execute the following command to start a single-machine 8-card data parallel training:

```bash
bash run.sh
```

Distributed training starts, and if the training process encounters an exception, such as a process exiting abnormally, the corresponding process is restarted and the training process can be resumed:
For example, if the Scheduler process exits abnormally in the middle of training, you can execute the following command to restart the Scheduler:

```bash
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/
export MS_ENABLE_RECOVERY=1                # Enable disaster recovery
export MS_RECOVERY_PATH=./recovery/        # Set the disaster recovery file save path

# Start 1 Scheduler process
export MS_WORKER_NUM=8              # Set the number of Worker processes in the cluster to 8
export MS_SCHED_HOST=127.0.0.1      # Set the Scheduler IP address to the local loop address
export MS_SCHED_PORT=8118           # Set the Scheduler Port
export MS_ROLE=MS_SCHED             # Set the startup process to MS_SCHED role
export MS_NODE_ID=sched             # Set this node Node ID to 'sched'
python ./train.py > device/scheduler.log 2>&1 &     # Start training scripts
```

Or execute the script:

```bash
bash recover.sh
```

The grouping of Worker and Scheduler is automatically restored.

Worker processes with abnormal exit are handled in a similar way (Note: Worker processes with abnormal exit need to wait for 30s before pulling up again to resume training, and before that, the Scheduler rejects Workers with the same node id from re-registering in order to prevent network jitter and malicious registrations).