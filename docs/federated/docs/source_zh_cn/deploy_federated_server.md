# 云侧部署

`Linux` `模型训练` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_zh_cn/deploy_federated_server.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本文档以LeNet网络为例，讲解如何使用MindSpore来部署联邦学习集群。

> 可以在[这里](https://gitee.com/mindspore/mindspore/tree/master/tests/st/fl/mobile)下载本文档中的完整Demo。

MindSpore Federated Learning Server集群物理架构如图所示：

![](./images/mindspore_federated_networking.png)

如上图所示，在联邦学习云侧集群中，有两种角色的MindSpore进程：`Federated Learning Scheduler`和`Federated Learning Server`:

- Federated Learning Scheduler

    `Scheduler`的作用主要有两点：

    1. 协助集群组网：在集群初始化阶段，由`Scheduler`负责收集`Server`信息，并达成集群一致性。`
    2. 开放管理面：支持用户通过`RESTful`接口对集群进行管理。

    在一个联邦学习任务中，只有一个`Scheduler`，与`Server`通过TCP协议通信。

- Federated Learning Server

    `Server`为执行联邦学习任务的主体，用于接收和解析来自端侧设备的数据，具有执行安全聚合、限时通信、模型存储等能力。在一个联邦学习任务中，`Server`可以有多个(用户可配置)，`Server`间通过TCP协议通信，对外开放HTTP端口用于端侧设备连接。

    在MindSpore联邦学习框架中，`Server`还支持弹性伸缩以及容灾，能够在训练任务不中断的情况下，动态调配硬件资源。

`Scheduler`和`Server`需部署在单网卡的服务器或者容器中，且处于相同网段。MindSpore自动获取首个可用IP地址作为`Server`地址。

## 准备环节

### 安装MindSpore

MindSpore联邦学习云侧集群支持在x86的CPU和GPU硬件平台上部署。执行[官网提供的命令](https://www.mindspore.cn/install)安装MindSpore最新版本。

## 定义模型

为了便于部署，MindSpore联邦学习的`Scheduler`和`Server`进程可以复用训练脚本，仅通过[参数配置](#id5)选择不同的启动方式。

本教程选择LeNet网络作为示例，具体的网络结构、损失函数和优化器定义请参考[LeNet网络样例脚本](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/lenet/lenet.py)。

## 参数配置

MindSpore联邦学习任务进程复用了训练脚本，用户只需要使用相同的脚本，并通过Python接口`set_fl_context`传递不同的参数，启动不同角色的MindSpore进程。参数配置说明请参考[API文档](https://www.mindspore.cn/federated/docs/zh-CN/master/federated_server.html#mindspore.context.set_fl_context)。

在确定参数配置后，用户需要在执行训练前调用`set_fl_context`接口，调用方式如下：

```python
import mindspore.context as context
...

enable_fl = True
server_mode = "FEDERATED_LEARNING"
ms_role = "MS_SERVER"
server_num = 4
scheduler_ip = "192.168.216.124"
scheduler_port = 6667
fl_server_port = 6668
fl_name = "LeNet"
scheduler_manage_port = 11202
config_file_path = "./config.json"

fl_ctx = {
    "enable_fl": enable_fl,
    "server_mode": server_mode,
    "ms_role": ms_role,
    "server_num": server_num,
    "scheduler_ip": scheduler_ip,
    "scheduler_port": scheduler_port,
    "fl_server_port": fl_server_port,
    "fl_name": fl_name,
    "scheduler_manage_port": scheduler_manage_port,
    "config_file_path": config_file_path
}
context.set_fl_context(**fl_ctx)
...

model.train()
```

本示例设置了训练任务的模式为`联邦学习`，训练进程角色为`Server`，需要启动`4`个`Server`才能完成集群组网，集群`Scheduler`的IP地址为`192.168.216.124`，集群`Scheduler`端口为`6667`，联邦学习`HTTP服务端口`为`6668`(由端侧设备连接)，任务名为`LeNet`，集群`Scheduler`管理端口为`11202`。

> 部分参数只在`Scheduler`用到，如scheduler_manage_port，部分参数只在`Server`用到，如fl_server_port，为了方便部署，可将这些参数配置统一传入，MindSpore会根据进程角色，读取不同的参数配置。

建议将参数配置通过Python `argparse`模块传入，以下是部分关键参数传入脚本的示例：

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--server_mode", type=str, default="FEDERATED_LEARNING")
parser.add_argument("--ms_role", type=str, default="MS_SERVER")
parser.add_argument("--server_num", type=int, default=4)
parser.add_argument("--scheduler_ip", type=str, default="192.168.216.124")
parser.add_argument("--scheduler_port", type=int, default=6667)
parser.add_argument("--fl_server_port", type=int, default=6668)
parser.add_argument("--fl_name", type=str, default="LeNet")
parser.add_argument("--scheduler_manage_port", type=int, default=11202)
parser.add_argument("--config_file_path", type=str, default="")

args, t = parser.parse_known_args()
server_mode = args.server_mode
ms_role = args.ms_role
server_num = args.server_num
scheduler_ip = args.scheduler_ip
scheduler_port = args.scheduler_port
fl_server_port = args.fl_server_port
fl_name = args.fl_name
scheduler_manage_port = args.scheduler_manage_port
config_file_path = args.config_file_path
```

> 每个Python脚本对应一个进程，若要在不同主机部署多个`Server`角色，则需要分别建立多个进程，可以通过shell指令配合Python的方式快速启动多`Server`。可参考**[示例](https://gitee.com/mindspore/mindspore/blob/master/tests/st/fl/mobile)**。
>
> 每个`Server`进程需要有一个集群内唯一标志`MS_NODE_ID`，需要通过环境变量设置此字段。本部署教程中，此变量已在[脚本run_mobile_server.py](https://gitee.com/mindspore/mindspore/blob/master/tests/st/fl/mobile/run_mobile_server.py)中设置。

## 启动集群

参考[示例](https://gitee.com/mindspore/mindspore/blob/master/tests/st/fl/mobile)，启动集群。参考示例关键目录结构如下：

```text
mobile/
├── config.json
├── finish_mobile.py
├── run_mobile_sched.py
├── run_mobile_server.py
├── src
│   └── model.py
└── test_mobile_lenet.py
```

- config.json：配置文件，用于安全能力配置，容灾等。
- finish_mobile.py：由于Server集群为常驻进程，使用本文件手动退出集群。
- run_mobile_sched.py：启动Scheduler。
- run_mobile_server.py：启动Server。
- model.py：网络模型。
- test_mobile_lenet.py：训练脚本

1. 启动Scheduler

    `run_mobile_sched.py`是为用户启动`Scheduler`而提供的Python脚本，并支持通过`argparse`传参修改配置。执行指令如下，代表启动本次联邦学习任务的`Scheduler`，其TCP端口为`6667`，联邦学习HTTP服务端口为`6668`，`Server`数量为`4`个，集群`Scheduler`管理端口为`11202`：

    ```sh
    python run_mobile_sched.py --scheduler_ip=192.168.216.124 --scheduler_port=6667 --fl_server_port=6668 --server_num=4 --scheduler_manage_port=11202
    ```

2. 启动Server

    `run_mobile_server.py`是为用户启动若干`Server`而提供的Python脚本，并支持通过`argparse`传参修改配置。执行指令如下，代表启动本次联邦学习任务的`Server`，其TCP端口为`6667`，联邦学习HTTP服务起始端口为`6668`，`Server`数量为`4`个，联邦学习任务正常进行需要的端侧设备数量为`8`个：

    ```sh
    python run_mobile_server.py --scheduler_ip=192.168.216.124 --scheduler_port=6667 --fl_server_port=6668 --server_num=4 --start_fl_job_threshold=8
    ```

    以上指令等价于启动了4个`Server`进程，每个`Server`的联邦学习服务端口分别为`6668`、`6669`、`6670`和`6671`，具体实现详见[脚本run_mobile_server.py](https://gitee.com/mindspore/mindspore/blob/master/tests/st/fl/mobile/run_mobile_server.py)。  

    > 若只想在单机部署`Scheduler`以及`Server`，只需将`scheduler_ip`配置项修改为`127.0.0.1`即可。

    若想让`Server`分布式部署在不同物理节点，可以使用`local_server_num`参数，代表在**本节点**需要执行的`Server`进程数量：

    ```sh
    # 在节点1启动3个Server进程
    python run_mobile_server.py --scheduler_ip=192.168.216.124 --scheduler_port=6667 --fl_server_port=6668 --server_num=4 --start_fl_job_threshold=8 --local_server_num=3
    ```

    ```sh
    # 在节点2启动1个Server进程
    python run_mobile_server.py --scheduler_ip=192.168.216.124 --scheduler_port=6667 --fl_server_port=6668 --server_num=4 --start_fl_job_threshold=8 --local_server_num=1
    ```

    看到日志中打印本行：

    ```sh
    Server started successfully.
    ```

    则说明启动成功。

    > 以上分布式部署的指令中，`server_num`都为4，这是因为此参数代表集群全局的`Server`数量，不应随着物理节点的数量而改变。对于不同节点上的`Server`来说，它们无需感知各自的IP地址，集群的一致性和节点发现都由`Scheduler`进行调度。

3. 停止联邦学习
    当前版本联邦学习集群为常驻进程，也可以执行`finish_mobile.py`脚本用于中途停止联邦学习服务器，执行如下指令来停止联邦学习集群，其中`scheduler_port`传参和启动服务器时的传参保持一致，代表停止此`Scheduler`对应的集群。

    ```sh
    python finish_mobile.py --scheduler_port=6667
    ```

    可看到结果：

    ```sh
    killed $PID1
    killed $PID2
    killed $PID3
    killed $PID4
    killed $PID5
    killed $PID6
    killed $PID7
    killed $PID8
    ```

    说明停止服务成功。

## 弹性伸缩

MindSpore联邦学习框架支持`Server`的弹性伸缩，对外通过`Scheduler`管理端口提供`RESTful`服务，使得用户在不中断训练任务的情况下，对硬件资源进行动态调度。目前MindSpore的弹性伸缩仅支持水平伸缩(Scale Out/In)，暂不支持垂直伸缩(Scale Up/Down)。在弹性伸缩场景下，必然会有`Server`进程的增加/减少。

以下详细描述用户如何通过RESTful原生接口，对集群扩容/缩容进行控制。

1. 扩容

    在集群启动后，向`Scheduler`发起扩容请求，这里使用`curl`指令构造`RESTful`扩容请求，代表集群需要扩容2个`Server`节点：

    ```sh
    curl -i -X POST \
    -H "Content-Type:application/json" \
    -d \
    '{
    "worker_num":0,
    "server_num":2
    }' \
    'http://192.168.216.124:11202/scaleout'
    ```

    需要拉起`2`个新的`Server`进程，并将`server_num`参数累加扩容的个数，从而保证全局组网信息的正确性，则扩容后，`server_num`的数量应为`6`，执行如下指令：

    ```sh
    python run_mobile_server.py --scheduler_ip=192.168.216.124 --scheduler_port=6667 --fl_server_port=6672 --server_num=6 --start_fl_job_threshold=8 --local_server_num=2
    ```

    此指令代表启动两个`Server`节点，联邦学习服务端口分别为`6672`和`6673`，总`Server`数量为`6`。

2. 缩容

    在集群启动后，向`Scheduler`发起缩容请求。由于缩容需要对具体节点进行操作，因此需要获取节点信息：

    ```sh
    curl -i -X GET \
    'http://192.168.216.124:11202/nodes'
    ```

    返回`json`格式的结果：

    ```json
    {
        "message": "Get nodes info successful.",
        "node_ids": [
            {
                "node_id": "40d56ffe-f8d1-4960-85fa-fdf88820402a",
                "rank_id": "3",
                "role": "SERVER"
            },
            {
                "node_id": "1ba06348-f2e2-4ad2-be83-0d41fcb53228",
                "rank_id": "2",
                "role": "SERVER"
            },
            {
                "node_id": "997967bb-c1ab-4916-8697-dcfaaf0354e5",
                "rank_id": "1",
                "role": "SERVER"
            },
            {
                "node_id": "4b8d5bdf-eafd-4f5c-8cae-79008f19298a",
                "rank_id": "0",
                "role": "SERVER"
            }
        ]
    }
    ```

    选择`Rank3`和`Rank2`进行缩容：

    ```sh
    curl -i -X POST \
    -H "Content-Type:application/json" \
    -d \
    '{
    "node_ids": ["40d56ffe-f8d1-4960-85fa-fdf88820402a", "1ba06348-f2e2-4ad2-be83-0d41fcb53228"]
    }' \
    'http://10.113.216.124:11202/scalein'
    ```

> - 在集群扩容/缩容成功后，训练任务会自动恢复，不需要用户进行额外干预。
>
> - 可以通过集群管理工具(如Kubernetes)创建或者释放`Server`资源。
>
> - 缩容后，被缩容节点进程不会退出，需要集群管理工具(如Kubernetes)释放`Server`资源或者执行`kill -15 $PID`来控制进程退出。

## 容灾

在MindSpore联邦学习集群中某节点下线后，可以保持集群在线而不退出训练任务，在该节点重新被启动后，可以恢复训练任务。目前MindSpore暂时支持Server节点的容灾(Server 0除外)。

想要支持容灾，config_file_path指定的config.json配置文件需要添加如下字段：

```json
{
    "recovery": {
        "storage_type": 1,
        "storge_file_path": "config.json"
    }
}
```

- recovery：有此字段则代表需要支持容灾。
- storage_type：持久化存储类型，目前只支持值为`1`，代表文件存储。
- storage_file_path：容灾恢复文件路径。

节点重新启动的指令类似扩容指令，在节点被手动下线之后，执行如下指令：

```sh
python run_mobile_server.py --scheduler_ip=192.168.216.124 --scheduler_port=6667 --fl_server_port=6673 --server_num=6 --start_fl_job_threshold=8 --local_server_num=1 --config_file_path=/home/config.json
```

此指令代表重新启动了`Server`，其联邦学习服务端口为`6673`。

> 在弹性伸缩命令下发成功后，在扩缩容业务执行完毕前，不支持容灾。
>
> 容灾后，重新启动节点的`MS_NODE_ID`变量需要和异常退出的节点保持一致，来保证能够恢复组网。

## 安全

MindSpore联邦学习框架支持`Server`的SSL安全认证，要开启安全认证，需要在启动命令加上enable_ssl=True，config_file_path指定的config.json配置文件需要添加如下字段：

```json
{
    "server_cert_path": "server.p12",
    "crl_path": "",
    "client_cert_path": "client.p12",
    "ca_cert_path": "ca.crt",
    "cert_expire_warning_time_in_day": 90,
    "cipher_list": "ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES256-GCM-SHA384:DHE-RSA-AES128-GCM-SHA256:DHE-DSS-AES128-GCM-SHA256:kEDH+AESGCM:ECDHE-RSA-AES128-SHA256:ECDHE-ECDSA-AES128-SHA256:ECDHE-RSA-AES128-SHA:ECDHE-ECDSA-AES128-SHA:ECDHE-RSA-AES256-SHA384:ECDHE-ECDSA-AES256-SHA384:ECDHE-RSA-AES256-SHA:ECDHE-ECDSA-AES256-SHA:DHE-RSA-AES128-SHA256:DHE-RSA-AES128-SHA:DHE-DSS-AES128-SHA256:DHE-RSA-AES256-SHA256:DHE-DSS-AES256-SHA:DHE-RSA-AES256-SHA:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!3DES:!MD5:!PSK",
    "connection_num":10000
}
```

- server_cert_path：服务端包含了证书和秘钥的密文的p12文件。
- crl_path：吊销列表的文件。
- client_cert_path：客户端包含了证书和秘钥的密文的p12文件。
- ca_cert_path：根证书。
- cipher_list：密码套件。
- cert_expire_warning_time_in_day：证书过期的告警时间。

p12文件中的秘钥为密文存储，在启动时需要传入密码，具体参数请参考Python API `mindspore.context.set_fl_context`中的`client_password`以及`server_password`字段。
