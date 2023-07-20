# 分布式故障恢复

<a href="https://gitee.com/mindspore/docs/blob/r2.0/tutorials/experts/source_zh_cn/parallel/fault_recover.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

## 概述

在进行分布式训练时，遇到故障是非常普遍的，类似于单卡训练，可以通过加载训练过程中保存的权重信息继续进行训练。区别于纯数据并行训练，当应用了模型并行后，权重是进行了切分的，卡与卡之间保存的权重信息可能不一致。
为了解决这个问题，一个方案是在保存权重checkpoint文件前，就将权重通过[AllGather](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0/parallel/communicate_ops.html#allgather) 算子进行汇聚，每张卡均存储一个完整的权重信息，这一个功能在[分布式训练模型参数保存和加载](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0/parallel/train_ascend.html#分布式训练模型参数保存和加载) 中已经介绍了。
但是，对于大模型来说，使用汇聚保存对各种资源的开销都过于巨大，因此，本文档介绍的是每张卡仅仅保存自身的权重信息的恢复方案。对于大模型来说，往往会同时应用上数据并行与模型并行，而数据并行的维度所划分的设备，它们持有的权重信息是完全一致的，这也为大模型提供了冗余的备份，本文档也将指出如何去获取这个冗余信息。
关于并行策略与权重的切片划分的关系，可以进行如下映射。关于数据并行，模型并行的概念，请参考[分布式训练](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0/parallel/train_ascend.html) 、关于优化器并行，请参考[优化器并行](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0/parallel/optimizer_parallel.html)。

- 数据并行 + 不开启优化器并行：并行通信域内的rank持有相同权重切片。
- 模型并行：并行通信域内的rank持有不同权重切片。
- 数据并行 + 开启优化器并行 + 优化器并行切满所有数据并行维度：并行通信域内的rank持有不同权重切片。
- 数据并行 + 开启优化器并行 + 优化器并行不切满所有数据并行维度：并行通信域内，优化器切分的通信域内的rank持有不同的权重切片，每个优化器切分的通信域之间持有相同的权重切片。

另外，需要注意的是，本文档介绍分布式故障恢复方案，需要在[下沉模式](https://www.mindspore.cn/docs/zh-CN/r2.0/design/overview.html#面向昇腾硬件的竞争力优化) 下使用。

>下载完整的样例代码：[distributed_training_transformer](https://gitee.com/mindspore/docs/tree/r2.0/docs/sample_code/distributed_training_transformer)

目录结构如下：

```text
└─sample_code
    ├─distribute_training_transformer
        ├── dataset.py
        ├── model.py
        ├── rank_table_8pcs.json
        ├── run_parallel_save_ckpt.sh
        ├── run_parallel_recover_ckpt.sh
        ├── parallel_save_ckpt_train.py
        └── parallel_recover_train.py
```

## 切片保存权重

保存切片的权重信息，仅仅需要在CheckpointConfig中配置integrated_save为False。同时，配置环境变量GROUP_INFO_FILE存储权重的冗余信息。

```bash
export GROUP_INFO_FILE=./group_info.pb
```

权重存储的代码部分如下，需要注意，训练时通过指定dataset_sink_mode为True以配置为下沉模式。

```python
import mindspore as ms
from mindspore.train import CheckpointConfig, ModelCheckpoint
from mindspore.nn import PipelineCell

def train():
    # model create
    # checkpoint save
    ckpt_config = CheckpointConfig(save_ckpt_steps=callback_size, keep_ckpt_max=4,
                                      integrated_save=False)
    ckpoint_cb = ModelCheckpoint(prefix="test", config=ckpt_config)
    callback = [ckpoint_cb]
    model.train(4, dataset, callbacks=callback, dataset_sink_mode=True)
```

## 加载权重继续训练

在上一步保存了权重切片后，在训练得到的目录下，以0卡目录为例，可以看到以下文件。

```text
└─ckpt_dir0
    ├── group_info.pb
    ├── test-1_77.ckpt
    └── train.log0
```

在train.log0中，可以看到当前训练后的loss值，类似如下。

```text
epoch: 1 step: 77, loss is 7.187697
epoch: 1 step: 77, loss is 6.612632
epoch: 1 step: 77, loss is 6.393444
epoch: 1 step: 77, loss is 6.271424
```

读取group_info.pb，可以获取到权重的冗余信息，该文件解析出来后将得到一个列表，该列表中的值为rank_id，表示这些列表中的rank_id对应的权重切片都是相同的，可以相互替换。
如下面的例子，0卡的group_info.pb解析出来后，发现0卡和4卡的权重切分是完全一致的，当0卡的checkpoint丢失时，可以直接复制4卡checkpoint作为0卡的checkpoint，进行恢复。

```python
import mindspore as ms
rank_list = ms.restore_group_info_list("./ckpt_dir0/group_info.pb")
print(rank_list) // [0, 4]
```

分布式的故障恢复，需要事先获取切分的信息，因而，需要先调用[model.build](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/train/mindspore.train.Model.html#mindspore.train.Model.build) 进行编译，继而再执行训练。

```python
import os
import mindspore as ms
def recover_train():
    # model create
    # checkpoint load
    if args_opt.ckpt_file:
        param_dict = ms.load_checkpoint(args_opt.ckpt_file)
        model.build(train_dataset=dataset, epoch=4)
        ms.load_param_into_net(net, param_dict)
    model.train(2, dataset, callbacks=callback, dataset_sink_mode=True)
```

## 准备环节

### 下载数据集

- [WMT14 En-Fr数据集下载](http://statmt.org/wmt14/test-full.tgz)，如果点击下载不成功，请尝试复制链接地址后下载。

使用`newstest2014-fren-ref.en.sgm`作为该任务的训练集合，合并且清洗该数据集。将数据集解压至`docs/sample_code/distributed_training_transformer`目录下。

### 预处理流程

执行下述代码进行数据的预处理过程，将会在当前目录下产生`output`目录，目录下将会生成`wmt14.en_fr.txt`和`wmt14.fr_en.txt`两个文件，文件中每行是一个法语和英语的句子对。我们将采用`wmt14.fr_en.txt`作为训练数据。

```python
python preprocess.py
```

### 配置分布式环境变量

在裸机环境（对比云上环境，即本地有Ascend 910 AI 处理器）进行分布式训练时，需要配置当前多卡环境的组网信息文件。如果使用华为云环境，因为云服务本身已经做好了配置，可以跳过本小节。

以Ascend 910 AI处理器为例，1个8卡环境的json配置文件示例如下，本样例将该配置文件命名为`rank_table_8pcs.json`。2卡环境配置可以参考样例代码中的`rank_table_2pcs.json`文件。

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "10.*.*.*",
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

其中需要根据实际训练环境修改的参数项有：

- `server_count`表示参与训练的机器数量。
- `server_id`表示当前机器的IP地址。
- `device_id`表示卡物理序号，即卡所在机器中的实际序号。
- `device_ip`表示集成网卡的IP地址，可以在当前机器执行指令`cat /etc/hccn.conf`，`address_x`的键值就是网卡IP地址。
- `rank_id`表示卡逻辑序号，固定从0开始编号。

### 调用集合通信库

MindSpore分布式并行训练的通信使用了华为集合通信库`Huawei Collective Communication Library`（以下简称HCCL），可以在Ascend AI处理器配套的软件包中找到。同时`mindspore.communication.management`中封装了HCCL提供的集合通信接口，方便用户配置分布式信息。
> HCCL实现了基于Ascend AI处理器的多机多卡通信，有一些使用限制，我们列出使用分布式服务常见的，详细的可以查看HCCL对应的使用文档。
>
> - 单机场景下支持1、2、4、8卡设备集群，多机场景下支持8*n卡设备集群。
> - 每台机器的0-3卡和4-7卡各为1个组网，2卡和4卡训练时卡必须相连且不支持跨组网创建集群。
> - 组建多机集群时需要保证各台机器使用同一交换机。
> - 服务器硬件架构及操作系统需要是SMP（Symmetrical Multi-Processing，对称多处理器）处理模式。

下面是调用集合通信库样例代码：

```python
import os
from mindspore.communication import init
import mindspore as ms

if __name__ == "__main__":
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=int(os.environ["DEVICE_ID"]))
    init()
    ...
```

其中，

- `mode=GRAPH_MODE`：使用分布式训练需要指定运行模式为图模式（PyNative模式不支持并行）。
- `device_id`：卡的物理序号，即卡所在机器中的实际序号。
- `init`：使能HCCL通信，并完成分布式训练初始化操作。

## 运行代码

在准备好数据和进入代码目录后，执行保存切片权重的训练脚本。

```bash
bash run_parallel_save_ckpt.sh DATASET_PATH
```

而后，执行故障恢复训练脚本。

```bash
bash run_parallel_recover_ckpt.sh DATASET_PATH
```

恢复训练结束后，查看loss如下，可以看到loss直接从6点多开始下降，说明加载成功了。

```text
epoch: 1 step: 77, loss is 6.465892
epoch: 1 step: 77, loss is 6.239279
```
