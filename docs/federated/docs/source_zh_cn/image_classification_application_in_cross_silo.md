# 实现一个云云联邦的图像分类应用(x86)

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_zh_cn/image_classification_application_in_cross_silo.md)

根据参与客户端的类型，联邦学习可分为云云联邦学习（cross-silo）和端云联邦学习（cross-device）。在云云联邦学习场景中，参与联邦学习的客户端是不同的组织（例如，医疗或金融）或地理分布的数据中心，即在多个数据孤岛上训练模型。在端云联邦学习场景中，参与的客户端为大量的移动或物联网设备。本框架将介绍如何在MindSpore Federated云云联邦框架上，使用网络LeNet实现一个图片分类应用。

启动云云联邦的图像分类应用的完整脚本可参考[这里](https://gitee.com/mindspore/federated/tree/master/example/cross_silo_femnist)。

## 下载数据集

本示例采用[leaf数据集](https://github.com/TalwalkarLab/leaf)中的联邦学习数据集`FEMNIST`，该数据集包含62个不同类别的手写数字和字母（数字0~9、26个小写字母、26个大写字母），图像大小为`28 x 28`像素，数据集包含3500个用户的手写数字和字母（最多可模拟3500个客户端参与联邦学习），总数据量为805263，平均每个用户包含数据量为226.83，所有用户数据量的方差为88.94。

可参考文档[端云联邦学习图像分类数据集处理](https://www.mindspore.cn/federated/docs/zh-CN/master/image_classfication_dataset_process.html)中步骤1~7获取图片形式的3500个用户数据集`3500_client_img`。

由于原始3500个用户数据集中每个用户数据量比较少，在云云联邦任务中会收敛太快，无法明显体现云云联邦框架的收敛效果，下面提供一个参考脚本，将指定数量的用户数据集合并为一个用户，以增加参与云云联邦任务的单个用户数据量，更好地模拟云云联邦框架实验。

```python
import os
import shutil


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def combine_users(root_data_path, new_data_path, raw_user_num, new_user_num):
    mkdir(new_data_path)
    user_list = os.listdir(root_data_path)
    num_per_user = int(raw_user_num / new_user_num)
    for i in range(new_user_num):
        print(
            "========================== combine the raw {}~{} users to the new user: dataset_{} ==========================".format(
                i * num_per_user, i * num_per_user + num_per_user - 1, i))
        new_user = "dataset_" + str(i)
        new_user_path = os.path.join(new_data_path, new_user)
        mkdir(new_user_path)
        for j in range(num_per_user):
            index = i * new_user_num + j
            user = user_list[index]
            user_path = os.path.join(root_data_path, user)
            tags = os.listdir(user_path)
            print("------------- process the raw user: {} -------------".format(user))
            for t in tags:
                tag_path = os.path.join(user_path, t)
                label_list = os.listdir(tag_path)
                new_tag_path = os.path.join(new_user_path, t)
                mkdir(new_tag_path)
                for label in label_list:
                    label_path = os.path.join(tag_path, label)
                    img_list = os.listdir(label_path)
                    new_label_path = os.path.join(new_tag_path, label)
                    mkdir(new_label_path)

                    for img in img_list:
                        img_path = os.path.join(label_path, img)
                        new_img_name = user + "_" + img
                        new_img_path = os.path.join(new_label_path, new_img_name)
                        shutil.copy(img_path, new_img_path)

if __name__ == "__main__":
    root_data_path = "cross_silo_femnist/femnist/3500_clients_img"
    new_data_path = "cross_silo_femnist/femnist/35_7_client_img"
    raw_user_num = 35
    new_user_num = 7
    combine_users(root_data_path, new_data_path, raw_user_num, new_user_num)
```

其中`root_data_path`为原始3500个用户数据集路径，`new_data_path`为合并后数据集的路径，`raw_user_num`指定用于合并的用户数据集总数（需<=3500），`new_user_num`用于设置将原始数据集合并为多少个用户。如示例代码中将从`cross_silo_femnist/femnist/3500_clients_img`中选取前35个用户，合并为7个用户数据集后存放在路径`cross_silo_femnist/femnist/35_7_client_img`（合并后的7个用户，每个用户包含原始的5个用户数据集）。

如下打印代表合并数据集成功：

```sh
========================== combine the raw 0~4 users to the new user: dataset_0 ==========================
------------- process the raw user: f1798_42 -------------
------------- process the raw user: f2149_81 -------------
------------- process the raw user: f4046_46 -------------
------------- process the raw user: f1093_13 -------------
------------- process the raw user: f1124_24 -------------
========================== combine the raw 5~9 users to the new user: dataset_1 ==========================
------------- process the raw user: f0586_11 -------------
------------- process the raw user: f0721_31 -------------
------------- process the raw user: f3527_33 -------------
------------- process the raw user: f0146_33 -------------
------------- process the raw user: f1272_09 -------------
========================== combine the raw 10~14 users to the new user: dataset_2 ==========================
------------- process the raw user: f0245_40 -------------
------------- process the raw user: f2363_77 -------------
------------- process the raw user: f3596_19 -------------
------------- process the raw user: f2418_82 -------------
------------- process the raw user: f2288_58 -------------
========================== combine the raw 15~19 users to the new user: dataset_3 ==========================
------------- process the raw user: f2249_75 -------------
------------- process the raw user: f3681_31 -------------
------------- process the raw user: f3766_48 -------------
------------- process the raw user: f0537_35 -------------
------------- process the raw user: f0614_14 -------------
========================== combine the raw 20~24 users to the new user: dataset_4 ==========================
------------- process the raw user: f2302_58 -------------
------------- process the raw user: f3472_19 -------------
------------- process the raw user: f3327_11 -------------
------------- process the raw user: f1892_07 -------------
------------- process the raw user: f3184_11 -------------
========================== combine the raw 25~29 users to the new user: dataset_5 ==========================
------------- process the raw user: f1692_18 -------------
------------- process the raw user: f1473_30 -------------
------------- process the raw user: f0909_04 -------------
------------- process the raw user: f1956_19 -------------
------------- process the raw user: f1234_26 -------------
========================== combine the raw 30~34 users to the new user: dataset_6 ==========================
------------- process the raw user: f0031_02 -------------
------------- process the raw user: f0300_24 -------------
------------- process the raw user: f4064_46 -------------
------------- process the raw user: f2439_77 -------------
------------- process the raw user: f1717_16 -------------
```

文件夹 `cross_silo_femnist/femnist/35_7_client_img`目录结构如下：

```text
35_7_client_img  # 将FeMnist数据集中35个用户合并为7个客户端数据（各包含5个用户数据）
├── dataset_0  # 客户端0的数据集
│   ├── train   # 训练数据集
│   │   ├── 0  # 存放类别0对应的图片数据
│   │   ├── 1  # 存放类别1对应的图片数据
│   │   │        ......
│   │   └── 61  # 存放类别61对应的图片数据
│   └── test  # 测试数据集，目录结构同train
│              ......
│
└── dataset_6  # 客户端6的数据集
    ├── train   # 训练数据集
    │   ├── 0  # 存放类别0对应的图片数据
    │   ├── 1  # 存放类别1对应的图片数据
    │   │        ......
    │   └── 61  # 存放类别61对应的图片数据
    └── test  # 测试数据集，目录结构同train
```

## 定义网络

我们选择相对简单的LeNet网络。LeNet网络不包括输入层的情况下，共有7层：2个卷积层、2个下采样层（池化层）、3个全连接层。每层都包含不同数量的训练参数，如下图所示：

![LeNet5](images/LeNet_5.jpg)

> 更多的LeNet网络的介绍不在此赘述，希望详细了解LeNet网络，可以查询<http://yann.lecun.com/exdb/lenet/>。

本任务使用的网络可参考脚本[test_cross_silo_femnist.py](https://gitee.com/mindspore/federated/blob/master/example/cross_silo_femnist/test_cross_silo_femnist.py)。

若想具体了解MindSpore中网络定义流程可参考[初学入门](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/quick_start.html#网络构建)。

## 启动云云联邦任务

### 安装MindSpore和MindSpore Federated

包括源码和下载发布版两种方式，支持CPU、GPU、Ascend硬件平台，根据硬件平台选择安装即可。安装步骤可参考[MindSpore安装指南](https://www.mindspore.cn/install)，[MindSpore Federated安装指南](https://www.mindspore.cn/federated/docs/zh-CN/master/federated_install.html)。

目前联邦学习框架只支持Linux环境中部署，cross-silo联邦学习框架需要MindSpore版本号>=1.5.0。

### 启动任务

参考[示例](https://gitee.com/mindspore/federated/tree/master/example/cross_silo_femnist)，启动集群。参考示例目录结构如下：

```text
cross_silo_femnist/
├── config.json # 配置文件
├── finish_cross_silo_femnist.py # 关闭云云联邦任务脚本
├── run_cross_silo_femnist_sched.py # 启动云云联邦scheduler脚本
├── run_cross_silo_femnist_server.py # 启动云云联邦server脚本
├── run_cross_silo_femnist_worker.py # 启动云云联邦worker脚本
├── run_cross_silo_femnist_worker_distributed.py # 启动云云联邦分布式训练worker脚本
└── test_cross_silo_femnist.py # 客户端使用的训练脚本
```

1. 启动Scheduler

   `run_cross_silo_femnist_sched.py`是为用户启动`Scheduler`而提供的Python脚本，并支持通过`argparse`传参修改配置。执行指令如下，代表启动本次联邦学习任务的`Scheduler`，其TCP端口为`5554`。

   ```sh
   python run_cross_silo_femnist_sched.py --scheduler_manage_address=127.0.0.1:5554
   ```

   打印如下代表启动成功：

   ```sh
   [INFO] FEDERATED(35566,7f4275895740,python):2022-10-09-15:23:22.450.205 [mindspore_federated/fl_arch/ccsrc/scheduler/scheduler.cc:35] Run] Scheduler started successfully.
   [INFO] FEDERATED(35566,7f41f259d700,python):2022-10-09-15:23:22.450.357 [mindspore_federated/fl_arch/ccsrc/common/communicator/http_request_handler.cc:90] Run] Start http server!
   ```

2. 启动Server

   `run_cross_silo_femnist_server.py`是为用户启动若干`Server`而提供的Python脚本，并支持通过`argparse`传参修改配置。执行指令如下，代表启动本次联邦学习任务的`Server`，其http起始端口为`5555`，`server`数量为`4`个。

   ```sh
    python run_cross_silo_femnist_server.py --local_server_num=4 --http_server_address=10.*.*.*:5555
   ```

   以上指令等价于启动了4个`Server`进程，每个`Server`的联邦学习服务端口分别为`5555`、`5556`、`5557`和`5558`。

3. 启动Worker

   `run_cross_silo_femnist_worker.py`是为用户启动若干`worker`而提供的Python脚本，并支持通过`argparse`传参修改配置。执行指令如下，代表启动本次联邦学习任务的`worker`，其http起始端口为`5555`，`worker`数量为`4`个：

   ```sh
   python run_cross_silo_femnist_worker.py --dataset_path=/data_nfs/code/fed_user_doc/federated/tests/st/cross_silo_femnist/35_7_client_img/ --http_server_address=10.*.*.*:5555
   ```

   当前云云联邦的`worker`节点支持单机多卡&多机多卡的分布式训练方式，`run_cross_silo_femnist_worker_distributed.py`是为用户启动`worker`节点的分布式训练而提供的Python脚本，并支持通过`argparse`传参修改配置。执行指令如下，代表启动本次联邦学习任务的分布式`worker`，其中`device_num`表示`worker`集群启动的进程数目，`run_distribute`表示启动集群的分布式训练，其http起始端口为`5555`，`worker`进程数量为`4`个：

   ```sh
   python run_cross_silo_femnist_worker_distributed.py --device_num=4 --run_distribute=True --dataset_path=/data_nfs/code/fed_user_doc/federated/tests/st/cross_silo_femnist/35_7_client_img/ --http_server_address=10.*.*.*:5555
   ```

当执行以上三个指令之后，进入当前目录下`worker_0`文件夹，通过指令`grep -rn "test acc" *`查看`worker_0`日志，可看到如下类似打印：

```sh
local epoch: 0, loss: 3.787421340711655, trian acc: 0.05342741935483871, test acc: 0.075
```

则说明云云联邦启动成功，`worker_0`正在训练，其他worker可通过类似方式查看。

若worker已分布式多卡训练的方式启动，进入当前目录下`worker_distributed/log_output/`文件夹，通过指令`grep -rn "test acc" *`查看`worker`分布式集群的日志，可看到如下类似打印：

```text
local epoch: 0, loss: 2.3467453340711655, trian acc: 0.06532451988877687, test acc: 0.076
```

以上脚本中参数配置说明请参考[yaml配置说明](https://www.mindspore.cn/federated/docs/zh-CN/master/horizontal/federated_server_yaml.html)。

### 日志查看

成功启动任务之后，会在当前目录`cross_silo_femnist`下生成相应日志文件，日志文件目录结构如下：

```text
cross_silo_femnist
├── scheduler
│   └── scheduler.log     # 运行scheduler过程中打印日志
├── server_0
│   └── server.log        # server_0运行过程中打印日志
├── server_1
│   └── server.log        # server_1运行过程中打印日志
├── server_2
│   └── server.log        # server_2运行过程中打印日志
├── server_3
│   └── server.log        # server_3运行过程中打印日志
├── worker_0
│   ├── ckpt              # 存放worker_0在每个联邦学习迭代结束时获取的聚合后的模型ckpt
│   │   ├── 0-fl-ms-bs32-0epoch.ckpt
│   │   ├── 0-fl-ms-bs32-1epoch.ckpt
│   │   │
│   │   │              ......
│   │   │
│   │   └── 0-fl-ms-bs32-19epoch.ckpt
│   └── worker.log        # 记录worker_0参与联邦学习任务过程中输出日志
└── worker_1
    ├── ckpt              # 存放worker_1在每个联邦学习迭代结束时获取的聚合后的模型ckpt
    │  ├── 1-fl-ms-bs32-0epoch.ckpt
    │  ├── 1-fl-ms-bs32-1epoch.ckpt
    │  │
    │  │                     ......
    │  │
    │  └── 1-fl-ms-bs32-19epoch.ckpt
    └── worker.log        # 记录worker_1参与联邦学习任务过程中输出日志
```

### 关闭任务

若想中途退出，则可用以下指令：

```sh
python finish_cross_silo_femnist.py --redis_port=2345
```

或者等待训练任务结束之后集群会自动退出，不需要手动关闭。

### 实验结果

- 使用数据：

  上面`下载数据集`部分生成的`35_7_client_img/`数据集

- 客户端本地训练epoch数：20

- 云云联邦学习总迭代数：20

- 实验结果（每个迭代聚合后模型在客户端的测试集上精度）

`worker_0`测试结果：

```sh
worker_0/worker.log:7409:local epoch: 0, loss: 3.787421340711655, trian acc: 0.05342741935483871, test acc: 0.075
worker_0/worker.log:14419:local epoch: 1, loss: 3.725699281115686, trian acc: 0.05342741935483871, test acc: 0.075
worker_0/worker.log:21429:local epoch: 2, loss: 3.5285709657335795, trian acc: 0.19556451612903225, test acc: 0.16875
worker_0/worker.log:28439:local epoch: 3, loss: 3.0393165519160608, trian acc: 0.4889112903225806, test acc: 0.4875
worker_0/worker.log:35449:local epoch: 4, loss: 2.575952764115026, trian acc: 0.6854838709677419, test acc: 0.60625
worker_0/worker.log:42459:local epoch: 5, loss: 2.2081101375296512, trian acc: 0.7782258064516129, test acc: 0.6875
worker_0/worker.log:49470:local epoch: 6, loss: 1.9229739431736557, trian acc: 0.8054435483870968, test acc: 0.69375
worker_0/worker.log:56480:local epoch: 7, loss: 1.7005576549999293, trian acc: 0.8296370967741935, test acc: 0.65625
worker_0/worker.log:63490:local epoch: 8, loss: 1.5248727620766704, trian acc: 0.8407258064516129, test acc: 0.6375
worker_0/worker.log:70500:local epoch: 9, loss: 1.3838803705352127, trian acc: 0.8568548387096774, test acc: 0.7
worker_0/worker.log:77510:local epoch: 10, loss: 1.265225578921041, trian acc: 0.8679435483870968, test acc: 0.7125
worker_0/worker.log:84520:local epoch: 11, loss: 1.167484122101638, trian acc: 0.8659274193548387, test acc: 0.70625
worker_0/worker.log:91530:local epoch: 12, loss: 1.082880981700859, trian acc: 0.8770161290322581, test acc: 0.65625
worker_0/worker.log:98540:local epoch: 13, loss: 1.0097520119572772, trian acc: 0.8840725806451613, test acc: 0.64375
worker_0/worker.log:105550:local epoch: 14, loss: 0.9469810053708015, trian acc: 0.9022177419354839, test acc: 0.7
worker_0/worker.log:112560:local epoch: 15, loss: 0.8907848935604703, trian acc: 0.9022177419354839, test acc: 0.6625
worker_0/worker.log:119570:local epoch: 16, loss: 0.8416629644123349, trian acc: 0.9082661290322581, test acc: 0.70625
worker_0/worker.log:126580:local epoch: 17, loss: 0.798475691030866, trian acc: 0.9122983870967742, test acc: 0.70625
worker_0/worker.log:133591:local epoch: 18, loss: 0.7599438544427897, trian acc: 0.9243951612903226, test acc: 0.6875
worker_0/worker.log:140599:local epoch: 19, loss: 0.7250227383907605, trian acc: 0.9294354838709677, test acc: 0.7125
```

其他客户端的测试结果基本相同，不再一一列出。