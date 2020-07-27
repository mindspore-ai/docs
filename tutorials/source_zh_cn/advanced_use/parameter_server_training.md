# Parameter Server训练

<!-- TOC -->

- [Parameter Server训练](#parameter_server训练)
    - [概述](#概述)
    - [准备工作](#准备工作)
        - [训练脚本准备](#训练脚本准备)
        - [参数设置](#参数设置)
        - [环境变量设置](#环境变量设置)
    - [执行训练](#执行训练)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/advanced_use/parameter_server_training.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

## 准备工作
以LeNet在Ascend 910上使用Parameter Server，并且配置单Worker，单Server训练为例：

### 训练脚本准备

参考<https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet>，了解如何训练一个LeNet网络。

### 参数设置

在本训练模式下，有以下两种调用接口方式以控制训练参数是否通过Parameter Server进行更新：
  
- 通过`mindspore.nn.Cell.set_param_ps()`对`nn.Cell`中所有权重递归设置
- 通过`mindspore.common.Parameter.set_param_ps()`对此权重进行设置

在原训练脚本基础上，设置LeNet模型所有权重通过Parameter Server训练：
```python
network = LeNet5(cfg.num_classes)
network.set_param_ps()
```

### 环境变量设置

Mindspore通过读取环境变量，控制Parameter Server训练，环境变量包括以下选项：

```
export MS_SERVER_NUM=1                # Server number
export MS_WORKER_NUM=1                # Worker number
export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
export MS_SCHED_POST=XXXX             # Scheduler port
export MS_ROLE=MS_SCHED               # The role of this process: MS_SCHED represents the scheduler, MS_WORKER represents the worker, MS_PSERVER represents the Server
```

## 执行训练

1. shell脚本

    提供Worker，Server和Scheduler三个角色对应的shell脚本，以启动训练：

    `Scheduler.sh`:
    ```bash
    #!/bin/bash
    export MS_SERVER_NUM=1
    export MS_WORKER_NUM=1
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX
    export MS_SCHED_POST=XXXX
    export MS_ROLE=MS_SCHED
    python train.py
    ```

    `Server.sh`:
    ```bash
    #!/bin/bash
    export MS_SERVER_NUM=1
    export MS_WORKER_NUM=1
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX
    export MS_SCHED_POST=XXXX
    export MS_ROLE=MS_PSERVER
    python train.py
    ```

    `Worker.sh`:
    ```bash
    #!/bin/bash
    export MS_SERVER_NUM=1
    export MS_WORKER_NUM=1
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX
    export MS_SCHED_POST=XXXX
    export MS_ROLE=MS_WORKER
    python train.py
    ```

    最后分别执行：
    ```bash
    sh Scheduler.sh > scheduler.log 2>&1 &
    sh Server.sh > server.log 2>&1 &
    sh Worker.sh > worker.log 2>&1 &
    ```
    启动训练

2. 查看结果

    查看`scheduler.log`中和Server与Worker通信日志：
    ```
    Bind to role=scheduler, id=1, ip=XXX.XXX.XXX.XXX, port=XXXX
    Assign rank=8 to node role=server, ip=XXX.XXX.XXX.XXX, port=XXXX
    Assign rank=9 to node role=worker, ip=XXX.XXX.XXX.XXX, port=XXXX
    the scheduler is connected to 1 workers and 1 servers
    ```
    说明Server、Worker与Scheduler通信建立成功。

    查看`worker.log`中训练结果：
    ```
    epoch: 1 step: 1, loss is 2.302287
    epoch: 1 step: 2, loss is 2.304071
    epoch: 1 step: 3, loss is 2.308778
    epoch: 1 step: 4, loss is 2.301943
    ...
    ```
