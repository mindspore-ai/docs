# 动态组网启动方式

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/dynamic_cluster.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

出于训练时的可靠性要求，MindSpore提供了**动态组网**特性，用户能够不依赖任何第三方库(OpenMPI)来启动Ascend/GPU/CPU分布式训练任务，并且训练脚本无需做任何修改。我们建议用户优先使用此种启动方式。用户可以点击[多卡启动方式](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/introduction.html#多卡启动方式)查看多卡启动方式在不同平台的支持情况。

OpenMPI在分布式训练的场景中，起到在Host侧同步数据以及进程间组网的功能；而MindSpore**动态组网**特性通过**复用Parameter Server模式训练架构**，取代了OpenMPI能力，可参考[Parameter Server模式](https://mindspore.cn/tutorials/experts/zh-CN/master/parallel/parameter_server_training.html)训练教程。

**动态组网**特性将多个MindSpore训练进程作为`Worker`启动，并且额外启动一个`Scheduler`负责组网和容灾恢复。用户只需对启动脚本做少量修改，即可执行分布式训练。

> 动态组网启动脚本能在多种硬件平台间快速迁移，无需对其进行额外修改。

## 注意事项

- 动态组网当前不支持`Pynative`模式。

## 环境变量

动态组网启动训练脚本前需要导出若干环境变量，如下表格所示：

<table align="center">
    <tr>
        <th align="left">环境变量</th>
        <th align="left">功能</th>
        <th align="left">类型</th>
        <th align="left">取值</th>
        <th align="left">说明</th>
    </tr>
    <tr>
        <td align="left">MS_ROLE</td>
        <td align="left">指定本进程角色。</td>
        <td align="left">String</td>
        <td align="left">
            <ul>
                <li>MS_SCHED: 代表Scheduler进程，一个训练任务只启动一个Scheduler，负责组网，容灾恢复等，<b>不会执行训练代码</b>。</li>
                <li>MS_WORKER: 代表Worker进程，一般设置分布式训练进程为此角色。</li>
                <li>MS_PSERVER: 代表Parameter Server进程，只有在Parameter Server模式下此角色生效，具体请参考<a link="(https://mindspore.cn/tutorials/experts/zh-CN/master/parallel/parameter_server_training.html">Parameter Server模式</a>。</li>
            </ul>
        </td>
        <td align="left">Worker和Parameter Server进程会向Scheduler进程注册从而完成组网。</td>
    </tr>
    <tr>
        <td align="left">MS_SCHED_HOST</td>
        <td align="left">指定Scheduler的IP地址。</td>
        <td align="left">String</td>
        <td align="left">合法的IP地址。</td>
        <td align="left">当前版本暂不支持IPv6地址。</td>
    </tr>
    <tr>
        <td align="left">MS_SCHED_PORT</td>
        <td align="left">指定Scheduler绑定端口号。</td>
        <td align="left">Integer</td>
        <td align="left">1024～65535范围内的端口号。</td>
        <td align="left"></td>
    </tr>
    <tr>
        <td align="left">MS_NODE_ID</td>
        <td align="left">指定本进程的ID，集群内唯一。</td>
        <td align="left">String</td>
        <td align="left">代表本进程的唯一ID，默认由MindSpore自动生成。</td>
        <td align="left">
            MS_NODE_ID在在以下情况需要设置，一般情况下无需设置，由MindSpore自动生成：
            <ul>
                <li>开启容灾场景：容灾恢复时需要获取当前进程ID，从而向Scheduler重新注册。</li>
                <li>开启GLOG日志重定向场景：为了保证各训练进程日志独立保存，需设置进程ID，作为日志保存路径后缀。</li>
                <li>指定进程rank id场景：用户可通过设置MS_NODE_ID为某个整数，来指定本进程的rank id。</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td align="left">MS_WORKER_NUM</td>
        <td align="left">指定角色为MS_WORKER的进程数量。</td>
        <td align="left">Integer</td>
        <td align="left">大于0的整数。</td>
        <td align="left">
            用户启动的Worker进程数量应当与此环境变量值相等。若小于此数值，组网失败；若大于此数值，Scheduler进程会根据Worker注册先后顺序完成组网，多余的Worker进程会启动失败。
        </td>
    </tr>
    <tr>
        <td align="left">MS_SERVER_NUM</td>
        <td align="left">指定角色为MS_PSERVER的进程数量。</td>
        <td align="left">Integer</td>
        <td align="left">大于0的整数。</td>
        <td align="left">只在Parameter Server训练模式下需要设置。</td>
    </tr>
    <tr>
        <td align="left">MS_ENABLE_RECOVERY</td>
        <td align="left">开启容灾。</td>
        <td align="left">Integer</td>
        <td align="left">1代表开启，0代表关闭。默认为0。</td>
        <td align="left"></td>
    </tr>
    <tr>
        <td align="left">MS_RECOVERY_PATH</td>
        <td align="left">持久化路径文件夹。</td>
        <td align="left">String</td>
        <td align="left">合法的用户目录。</td>
        <td align="left">Worker和Scheduler进程在执行过程中会进行必要的持久化，如用于恢复组网的节点信息以及训练业务中间状态等，并通过文件保存。</td>
    </tr>
    <tr>
        <td align="left">MS_HCCL_CM_INIT</td>
        <td align="left">是否使用CM方式初始化HCCL。</td>
        <td align="left">Integer</td>
        <td align="left">1代表是，0代表否。默认为0。</td>
        <td align="left">此环境变量只在<b>Ascend硬件平台并且通信域数量较多</b>的情况下建议开启。开启此环境变量后，能够降低HCCL集合通信库的内存占用，并且训练任务执行方式与`rank table`启动方式相同。</td>
    </tr>
</table>

> 以上环境变量在各进程启动前都需设置且`MS_SCHED_HOST`，`MS_SCHED_PORT`，`MS_WORKER_NUM`内容保持一致，否则会由于各进程配置不一致导致组网失败。

## 执行训练任务

由于**动态组网**启动脚本在各硬件平台下能够保持一致，下面仅以GPU硬件平台下使用8卡分布式训练为例，演示如何编写启动脚本：

> 样例的运行目录：[distributed_training](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training)。

### 1. 准备Python训练脚本：

```python
import mindspore as ms
from mindspore.train import CheckpointConfig, ModelCheckpoint
from mindspore.communication import init

if __name__ == "__main__":
    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
    init()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    ...
```

其中，

- `mode=GRAPH_MODE`：使用分布式训练需要指定运行模式为图模式（当前版本**动态组网**特性暂不支持PyNative模式）。
- `init()`：初始化组网，根据`set_context`接口中指定后端，初始化集合通信库（此案例下为NCCL），完成分布式训练初始化操作。
- `ms.ParallelMode.DATA_PARALLEL`：设置训练模式为数据并行模式。

动态组网还支持**安全加密通道**特性，支持`TLS/SSL`协议，满足用户的安全性需求。默认情况下，安全加密通道是关闭的，若需要开启，则通过`set_ps_context`正确配置安全加密通道后，才能调用init()，否则初始化组网会失败。若想使用安全加密通道，请配置：

`set_ps_context(config_file_path="/path/to/config_file.json", enable_ssl=True, client_password="123456", server_password="123456")`

> 详细参数配置说明请参考Python API [mindspore.set_ps_context](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_ps_context.html#mindspore.set_ps_context)，以及本文档[安全认证](#安全认证)章节。

### 2. 准备启动脚本

#### 单机多卡

单机多卡启动脚本内容`run_gpu_cluster.sh`如下，在启动Worker和Scheduler之前，需要添加相关环境变量设置：

```bash
#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster.sh DATA_PATH"
echo "For example: bash run_gpu_cluster.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu.py ./resnet.py ./device
cd ./device
echo "start training"

# 循环启动8个Worker训练进程
for((i=0;i<8;i++));
do
    export MS_WORKER_NUM=8          # 设置集群中Worker进程数量为8
    export MS_SCHED_HOST=127.0.0.1  # 设置Scheduler IP地址为本地环路地址
    export MS_SCHED_PORT=8118       # 设置Scheduler端口
    export MS_ROLE=MS_WORKER        # 设置启动的进程为MS_WORKER角色
    export MS_NODE_ID=$i                      # 设置进程id，可选
    pytest -s -v ./resnet50_distributed_training_gpu.py > worker_$i.log 2>&1 &                             # 启动训练脚本
done

# 启动1个Scheduler进程
export MS_WORKER_NUM=8              # 设置集群中Worker进程数量为8
export MS_SCHED_HOST=127.0.0.1      # 设置Scheduler IP地址为本地环路地址
export MS_SCHED_PORT=8118           # 设置Scheduler端口
export MS_ROLE=MS_SCHED             # 设置启动的进程为MS_SCHED角色
pytest -s -v ./resnet50_distributed_training_gpu.py > scheduler.log 2>&1 &     # 启动训练脚本
```

> Scheduler和Worker进程的训练脚本内容和启动方式完全一致，这是因为在MindSpore已经差异化处理了两种角色内部流程。用户只需按照普通的训练方式拉起进程即可，无需按照角色修改Python代码。这是动态组网启动脚本在多硬件平台能够保持一致的原因之一。

执行如下指令，即可执行单机8卡分布式训练：

```bash
./run_gpu_cluster.sh /path/to/dataset/
```

#### 多机多卡

多机训练场景下，需拆分启动脚本。下面以执行2机8卡训练，每台机器执行启动4 Worker为例：

脚本`run_gpu_cluster_1.sh`在节点1上启动1`Scheduler`和`Worker1`到`Worker4`：

```bash
#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster.sh DATA_PATH"
echo "For example: bash run_gpu_cluster.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu.py ./resnet.py ./device
cd ./device
echo "start training"

# 循环启动Worker1到Worker4，4个Worker训练进程
for((i=0;i<4;i++));
do
    export MS_WORKER_NUM=8                    # 设置集群中Worker进程总数为8（包括其他器节点进程）
    export MS_SCHED_HOST=<node_1 ip address>  # 设置Scheduler IP地址为节点1 IP地址
    export MS_SCHED_PORT=8118                 # 设置Scheduler端口
    export MS_ROLE=MS_WORKER                  # 设置启动的进程为MS_WORKER角色
    export MS_NODE_ID=$i                      # 设置进程id，可选
    pytest -s -v ./resnet50_distributed_training_gpu.py > worker_$i.log 2>&1 &                                       # 启动训练脚本
done

# 在节点1启动1个Scheduler进程
export MS_WORKER_NUM=8                        # 设置集群中Worker进程总数为8（包括其他器节点进程）
export MS_SCHED_HOST=<node_1 ip address>      # 设置Scheduler IP地址为节点1 IP地址
export MS_SCHED_PORT=8118                     # 设置Scheduler端口
export MS_ROLE=MS_SCHED                       # 设置启动的进程为MS_SCHED角色
pytest -s -v ./resnet50_distributed_training_gpu.py > scheduler.log 2>&1 &     # 启动训练脚本
```

脚本`run_gpu_cluster_2.sh`在节点2上启动`Worker5`到`Worker8`（无需执行Scheduler）：

```bash
#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster.sh DATA_PATH"
echo "For example: bash run_gpu_cluster.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu.py ./resnet.py ./device
cd ./device
echo "start training"

# 循环启动Worker5到Worker8，4个Worker训练进程
for((i=4;i<8;i++));
do
    export MS_WORKER_NUM=8                    # 设置集群中Worker进程总数为8（包括其他器节点进程）
    export MS_SCHED_HOST=<node_1 ip address>  # 设置Scheduler IP地址为节点1 IP地址
    export MS_SCHED_PORT=8118                 # 设置Scheduler端口
    export MS_ROLE=MS_WORKER                  # 设置启动的进程为MS_WORKER角色
    export MS_NODE_ID=$i                      # 设置进程id，可选
    pytest -s -v ./resnet50_distributed_training_gpu.py > worker_$i.log 2>&1 &                                       # 启动训练脚本
done
```

> 多机任务`MS_WORKER_NUM`应当为集群中Worker节点总数。
> 节点间网络需保持连通，可使用`telnet <scheduler ip> <scheduler port>`指令测试本节点是否和已启动的Scheduler节点连通。

在节点1执行：

```bash
./run_gpu_cluster_1.sh /path/to/dataset/
```

在节点2执行：

```bash
./run_gpu_cluster_2.sh /path/to/dataset/
```

即可执行2机8卡分布式训练任务。

> 上述启动脚本在`Ascend`以及`CPU`硬件平台下保持一致，只需对Python训练脚本中`device_target`等硬件相关代码修改即可执行动态组网分布式训练。

### 3. 执行结果

脚本会在后台运行，日志文件会保存到当前目录下，共跑了10个epoch，每个epoch有234个step，关于Loss部分结果保存在worker_*.log中。将loss值grep出来后，示例如下：

```text
epoch: 1 step: 234, loss is 2.0084016
epoch: 2 step: 234, loss is 1.6407638
epoch: 3 step: 234, loss is 1.6164391
epoch: 4 step: 234, loss is 1.6838071
epoch: 5 step: 234, loss is 1.6320667
epoch: 6 step: 234, loss is 1.3098773
epoch: 7 step: 234, loss is 1.3515002
epoch: 8 step: 234, loss is 1.2943741
epoch: 9 step: 234, loss is 1.2316195
epoch: 10 step: 234, loss is 1.1533381
```

## 容灾恢复

模型训练对分布式训练架构的可靠性、可服务性要求比较高，MindSpore支持数据并行下容灾恢复，多卡数据并行训练场景集群(多个Worker和1个Scheduler)中存在进程异常退出，被重新拉起后，训练任务继续能正常执行。

场景约束：
在图模式下，采用`MindData`进行数据下沉模式训练，开启数据并行模式，采用上述的非`OpenMPI`的方式拉起Worker进程。

在上述场景下，训练过程中如果有节点挂掉，保证在相同的环境变量（`MS_ENABLE_RECOVERY` 和 `MS_RECOVERY_PATH`）下，重新拉起对应进程对应的脚本后训练可继续，并且不影响精度收敛。

1） 开启容灾：

通过环境变量开启容灾：

```bash
export MS_ENABLE_RECOVERY=1                 # 开启容灾
export MS_RECOVERY_PATH=/path/to/recovery/  # 配置持久化路径文件
```

2）配置checkpoint保存间隔，样例如下：

```python
from mindspore.train import ModelCheckpoint, CheckpointConfig

ckptconfig = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=5)
ckpoint_cb = ModelCheckpoint(prefix='train', directory="./ckpt_of_rank_/"+str(get_rank()), config=ckptconfig)
```

每个Worker都开启保存checkpoint，并用不同的路径（如上述样例中的directory的设置使用了rank id，保证路径不会相同），防止同名checkpoint保存冲突。checkpoint用于异常进程恢复和正常进程回滚，训练的回滚是指集群中各个Worker都恢复到最新的checkpoint对应的状态，同时数据侧也回退到对应的step，然后继续训练。保存checkpoint的间隔是可配置的，这个间隔决定了容灾恢复的粒度，间隔越小，恢复到上次保存checkpoint所回退的step数就越小，但保存checkpoint频繁也可能会影响训练效率，间隔越大则效果相反。keep_checkpoint_max至少设置为2(防止checkpoint保存失败)。

> 样例的运行目录：[distributed_training](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training)。

涉及到的脚本有`run_gpu_cluster_recovery.sh`、`resnet50_distributed_training_gpu_recovery.py`、`resnet.py`。脚本内容`run_gpu_cluster_recovery.sh`如下：

```bash
#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster_recovery.sh DATA_PATH"
echo "For example: bash run_gpu_cluster_recovery.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

export MS_ENABLE_RECOVERY=1                # 开启容灾
export MS_RECOVERY_PATH=/path/to/recovery/ # 配置持久化路径文件夹

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu_recovery.py ./resnet.py ./device
cd ./device
echo "start training"

# 启动1个Scheduler进程
export MS_WORKER_NUM=8              # 设置集群中Worker进程数量为8
export MS_SCHED_HOST=127.0.0.1      # 设置Scheduler IP地址为本地环路地址
export MS_SCHED_PORT=8118           # 设置Scheduler端口
export MS_ROLE=MS_SCHED             # 设置启动的进程为MS_SCHED角色
export MS_NODE_ID=sched             # 设置本节点Node ID为'sched'
pytest -s -v ./resnet50_distributed_training_gpu_recovery.py > scheduler.log 2>&1 &

# 循环启动8个Worker训练进程
for((i=0;i<8;i++));
do
    export MS_WORKER_NUM=8              # 设置集群中Worker进程数量为8
    export MS_SCHED_HOST=127.0.0.1      # 设置Scheduler IP地址为本地环路地址
    export MS_SCHED_PORT=8118           # 设置Scheduler端口
    export MS_ROLE=MS_WORKER            # 设置启动的进程为MS_WORKER角色
    export MS_NODE_ID=worker_$i         # 设置本节点Node ID为'worker_$i'
    pytest -s -v ./resnet50_distributed_training_gpu_recovery.py > worker_$i.log 2>&1 &
done
```

在启动Worker和Scheduler之前，需要添加相关环境变量设置，如Scheduler的IP和Port，当前进程的角色是Worker还是Scheduler。

执行下面的命令即可启动一个单机8卡的数据并行训练

```bash
bash run_gpu_cluster_recovery.sh /path/to/recovery/
```

分布式训练开始，若训练过程中遇到异常，如进程异常退出，然后再重新启动对应的进程，训练流程即可恢复：
例如训练中途Scheduler进程异常退出，可执行下列命令重新启动Scheduler：

```bash
export DATA_PATH=YOUR_DATA_PATH
export MS_ENABLE_RECOVERY=1                # 开启容灾功能
export MS_RECOVERY_PATH=/path/to/recovery/ # 设置容灾文件保存路径

cd ./device

# 启动1个Scheduler进程
export MS_WORKER_NUM=8              # 设置集群中Worker进程数量为8
export MS_SCHED_HOST=127.0.0.1      # 设置Scheduler IP地址为本地环路地址
export MS_SCHED_PORT=8118           # 设置Scheduler端口
export MS_ROLE=MS_SCHED             # 设置启动的进程为MS_SCHED角色
export MS_NODE_ID=sched             # 设置本节点Node ID为'sched'
pytest -s -v ./resnet50_distributed_training_gpu_recovery.py > scheduler.log 2>&1 &
```

Worker和Scheduler的组网会自动恢复。

Worker进程出现异常退出处理方式类似(注：Worker进程出现异常退出，需要等30s后再拉起才能恢复训练，在这之前，Scheduler为了防止网络抖动和恶意注册，拒绝相同node id的Worker再次注册)。

## 安全认证

要支持节点/进程间的SSL安全认证，要开启安全认证，通过Python API `mindspore.set_ps_context`配置`enable_ssl=True`(不传入时默认为False，表示不启用SSL安全认证)，config_file_path指定的config.json配置文件需要添加如下字段：

```json
{
  "server_cert_path": "server.p12",
  "crl_path": "",
  "client_cert_path": "client.p12",
  "ca_cert_path": "ca.crt",
  "cipher_list": "ECDHE-R SA-AES128-GCM-SHA256:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES256-GCM-SHA384:DHE-RSA-AES128-GCM-SHA256:DHE-DSS-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384:DHE-DSS-AES256-GCM-SHA384:DHE-PSK-AES128-GCM-SHA256:DHE-PSK-AES256-GCM-SHA384:DHE-PSK-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-PSK-CHACHA20-POLY1305:DHE-RSA-AES128-CCM:DHE-RSA-AES256-CCM:DHE-RSA-CHACHA20-POLY1305:DHE-PSK-AES128-CCM:DHE-PSK-AES256-CCM:ECDHE-ECDSA-AES128-CCM:ECDHE-ECDSA-AES256-CCM:ECDHE-ECDSA-CHACHA20-POLY1305",
  "cert_expire_warning_time_in_day": 90
}
```

- server_cert_path: 服务端包含了证书和秘钥的密文的p12文件（SSL专用证书文件）路径。
- crl_path: 吊销列表（用于区分无效不可信证书和有效可信证书）的文件路径。
- client_cert_path: 客户端包含了证书和秘钥的密文的p12文件（SSL专用证书文件）路径。
- ca_cert_path: 根证书路径。
- cipher_list: 密码套件（支持的SSL加密类型列表）。
- cert_expire_warning_time_in_day: 证书过期的告警时间。

p12文件中的秘钥为密文存储，在启动时需要传入密码，具体参数请参考Python API [mindspore.set_ps_context](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_ps_context.html#mindspore.set_ps_context)中的`client_password`以及`server_password`字段。
