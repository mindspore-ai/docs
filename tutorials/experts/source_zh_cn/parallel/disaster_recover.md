# 动态组网场景下故障恢复

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.2/tutorials/experts/source_zh_cn/parallel/disaster_recover.md)

## 概述

模型训练对分布式训练架构的可靠性、可服务性要求比较高，MindSpore动态组网启动方式支持数据并行下容灾恢复，多卡数据并行训练场景集群(多个Worker和1个Scheduler)中存在进程异常退出，被重新拉起后，训练任务继续能正常执行。

具体来说，在图模式下，采用数据下沉模式进行训练，并开启数据并行模式，采用动态组网方式启动训练集群后，训练过程中如果有进程异常退出，保证在相同的环境变量（`MS_ENABLE_RECOVERY` 和 `MS_RECOVERY_PATH`）下，重新拉起对应进程对应的脚本后训练可继续，并且不影响精度收敛。

> 动态组网场景下的容灾恢复仅支持GPU，需要在Graph模式下运行。

更多详细说明请查看[环境变量目录](https://www.mindspore.cn/docs/zh-CN/r2.2/note/env_var_list.html)中的动态组网环境变量。

## 操作实践

下面以Ascend为例进行操作说明：

### 样例代码说明

> 下载完整的样例代码：[disaster_recover](https://gitee.com/mindspore/docs/tree/r2.2/docs/sample_code/disaster_recover)。

目录结构如下：

```text
└─ sample_code
    ├─ disaster_recover
       ├── train.py
       ├── run.sh
       └── recover.sh
    ...
```

其中，`train.py`是定义网络结构和训练过程的脚本。`run.sh`是执行脚本，`recover.sh`是节点故障后的恢复脚本。

### 网络结构

网络结构和数据集加载与[动态组网启动](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.2/parallel/dynamic_cluster.html)中的示例一致。

### 定义训练过程

```python
import mindspore as ms
from mindspore import nn, train

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()
loss_cb = train.LossMonitor(20)
# 配置保存checkpoint的间隔，以及最大保存数量
ckpt_config = train.CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=5)
# 配置checkpoint保存路径，每个进程用不同的路径
ckpoint_cb = train.ModelCheckpoint(prefix='train', directory="./ckpt_of_rank/"+str(get_rank()), config=ckpt_config)
model = ms.Model(net, loss_fn=loss_fn, optimizer=optimizer)
model.train(10, data_set, callbacks=[loss_cb, ckpoint_cb])
```

每个Worker都开启保存checkpoint，并用不同的路径（如上述样例中的directory的设置使用了rank id，保证路径不会相同），防止同名checkpoint保存冲突。checkpoint用于异常进程恢复和正常进程回滚，训练的回滚是指集群中各个Worker都恢复到最新的checkpoint对应的状态，同时数据侧也回退到对应的step，然后继续训练。

保存checkpoint的间隔是可配置的，这个间隔决定了容灾恢复的粒度，间隔越小，恢复到上次保存checkpoint所回退的step数就越小，但保存checkpoint频繁也可能会影响训练效率，间隔越大则效果相反。keep_checkpoint_max至少设置为2(防止checkpoint保存失败)。

### 准备启动脚本

脚本内容`run.sh`如下，增加容灾恢复相关的环境变量：

```bash
EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

export MS_ENABLE_RECOVERY=1                # 开启容灾
export MS_RECOVERY_PATH=./recovery/        # 设置容灾文件保存路径

rm -rf device
mkdir device
echo "start training"

# 循环启动8个Worker训练进程
for((i=0;i<8;i++));
do
    export MS_WORKER_NUM=8          # 设置集群中Worker进程数量为8
    export MS_SCHED_HOST=127.0.0.1  # 设置Scheduler IP地址为本地环路地址
    export MS_SCHED_PORT=8118       # 设置Scheduler端口
    export MS_ROLE=MS_WORKER        # 设置启动的进程为MS_WORKER角色
    export MS_NODE_ID=$i            # 设置进程id，可选
    python ./train.py > device/worker_$i.log 2>&1 &     # 启动训练脚本
done

# 启动1个Scheduler进程
export MS_WORKER_NUM=8              # 设置集群中Worker进程数量为8
export MS_SCHED_HOST=127.0.0.1      # 设置Scheduler IP地址为本地环路地址
export MS_SCHED_PORT=8118           # 设置Scheduler端口
export MS_ROLE=MS_SCHED             # 设置启动的进程为MS_SCHED角色
python ./train.py > device/scheduler.log 2>&1 &     # 启动训练脚本
```

其中环境变量`MS_ENABLE_RECOVERY=1`表示开启容灾，`MS_RECOVERY_PATH=./recovery/`表示配置存放持久化文件的路径。

在启动Worker和Scheduler之前，需要添加相关环境变量设置，比如：

- `MS_WORKER_NUM=8`：配置Worker进程数量为8。
- `MS_SCHED_HOST=127.0.0.1`：配置Scheduler IP地址为127.0.0.1。
- `MS_SCHED_PORT=8118`：配置Scheduler的端口号为8118。
- `MS_ROLE=MS_WORKER`：配置当前进程的角色，`MS_WORKER`代表角色是Worker，`MS_SCHED`代表角色是Scheduler。

执行下面的命令即可启动一个单机8卡的数据并行训练：

```bash
bash run.sh
```

分布式训练开始，若训练过程中遇到异常，如进程异常退出，然后再重新启动对应的进程，训练流程即可恢复：
例如训练中途Scheduler进程异常退出，可执行下列命令重新启动Scheduler：

```bash
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/
export MS_ENABLE_RECOVERY=1                # 开启容灾功能
export MS_RECOVERY_PATH=./recovery/        # 设置容灾文件保存路径

# 启动1个Scheduler进程
export MS_WORKER_NUM=8              # 设置集群中Worker进程数量为8
export MS_SCHED_HOST=127.0.0.1      # 设置Scheduler IP地址为本地环路地址
export MS_SCHED_PORT=8118           # 设置Scheduler端口
export MS_ROLE=MS_SCHED             # 设置启动的进程为MS_SCHED角色
export MS_NODE_ID=sched             # 设置本节点Node ID为'sched'
python ./train.py > device/scheduler.log 2>&1 &     # 启动训练脚本
```

或者执行脚本：

```bash
bash recover.sh
```

Worker和Scheduler的组网会自动恢复。

Worker进程出现异常退出处理方式类似(注：Worker进程出现异常退出，需要等30s后再拉起才能恢复训练，在这之前，Scheduler为了防止网络抖动和恶意注册，拒绝相同node id的Worker再次注册)。
