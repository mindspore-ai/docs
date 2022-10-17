# 分布式弹性训练与推理（Semi-auto/Auto Parallel模式）

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/resilience_train_and_predict.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

### 背景

在使用MindSpore进行分布式训练时，常常需要对训练得到的分布式Checkpoint进行转换以进行下一步工作，如推理、微调、多阶段训练等。本教程将介绍如何将分布式训练得到的Checkpoint进行转换以开展分布式策略与集群卡数改变的弹性训练与推理。
本功能仅支持semi_auto_parallel/auto_parallel模式，暂时不支持流水线并行维度的转换。

### 使用场景

如果您遇到如下场景，需要参考本教程操作，进行弹性训练与推理：

场景1：M卡训练，N卡微调训练，M与N可以没有倍数关系。
场景2：训练分为多阶段，每个阶段的集群大小不一样。
场景3：M卡训练，N卡推理，M与N可以没有倍数关系。
场景4：需要对网络的切分策略进行变更。

以在8卡上训练，并在4卡上微调为例，整体操作流程如下：

1. 执行训练，配置模型参数切分策略文件存储位置，自动生成Checkpoint文件和模型参数切分策略文件。

2. 编译微调网络，配置分布式策略文件存储位置，自动生成模型参数切分策略文件。

3. 用户对依据训练与推理涉及到的策略文件对保存的Checkpoint文件进行转换。

4. 编译微调网络后，加载转换得到的分布式Checkpoint文件。

5. 执行微调网络。

需要注意，加载分布式的Checkpoint，要求对网络进行编译后才可以加载。

> 数据集下载，请参考分布式并行训练Transformer模型教程中的[准备环节](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/transformer.html#准备环节)
>
>您可以在这里下载完整的样例代码：
>
><https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_resilience_training>

## 对分布式Checkpoint文件进行转换

### 整体流程

首先，执行分布式训练，并行模式设置为`semi_auto_parallel`/`auto_parallel`，同时通过调用`set_auto_parallel_context`接口自定义`strategy_ckpt_save_file`参数配置模型切分策略文件存储路径,
训练一段时间后，调用存储Checkpoint的callback函数，将分布式的Checkpoint存储下来。而后编译新的卡数/切分策略下的网络，生成目标网络的模型切分策略文件，调用分布式Checkpoint转换的接口进行分布式Checkpoint的转换。

### 执行分布式训练

定义网络，进行分布式的初始化，获取设备数与卡号，对于非流水线并行的情况下，每张卡的切分策略文件内容均是一致的，因此只对0卡调用`set_auto_parallel_context(strategy_ckpt_save_file="../src_strategy.ckpt")`保存策略文件即可。
添加保存Checkpoint的回调函数，首先定义Checkpoint存储相关的配置对象`CheckpointConfig`，注意`integrated_save`配置为`False`，意味着不对分布式训练的权重做聚合保存，以适应大模型下的内存开销。
而后定义保存Checkpoint的回调函数`ModelCheckpoint`。最后，调用`model.train`执行训练。
关于分布式训练的基本使用方法，请参考[分布式训练Ascend](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_ascend.html)
或者[分布式训练GPU](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_gpu.html) 。

```python
import mindspore as ms
import mindspore.communication as D
D.init()
device_num = D.get_group_size()
rank_id = D.get_rank()
net = Net()
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
if rank_id == 0:
    ms.set_auto_parallel_context(strategy_ckpt_save_file="../src_strategy.ckpt")
opt = Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())
model = ms.Model(net, optimizer=opt)
ckpt_config = ms.CheckpointConfig(save_checkpoint_steps=callback_size, keep_checkpoint_max=1,
                                  integrated_save=False)
ckpoint_cb = ms.ModelCheckpoint(prefix="src_checkpoint",
                                directory = "../src_checkpoints/rank_{}".format(rank_id),
                                config=ckpt_config)
callback = [ms.TimeMonitor(callback_size), ms.LossMonitor(callback_size), ckpoint_cb]
model.train(2, dataset, callbacks=callback, dataset_sink_mode=True)
```

其中，

- `dataset`：MindData对象，需要提前构造好以给入`model.train`。

示例里面执行8卡训练脚本执行命令为：

```bash
bash run_train_8p.sh ../output/wmt14.en_fr.txt
```

执行后，将会生成源Checkpoint文件目录以及源切分策略文件:

```text
src_checkpoints/
src_strategy.ckpt
```

### 对分布式Checkpoint进行转换

#### 对目标网络执行编译

进行分布式的Checkpoint的转换，依赖于原始的分布式策略文件与目标的分布式策略文件，执行原始的策略下的网络训练时，已经将分布式策略文件存储下来了，因此需要另外获取到目标策略下的分布式策略文件。
通过对目标策略的网络执行编译，即可获取到目标策略网络的分布式策略文件。通过`model.build`接口既可以单独对网络执行编译。

```python
import mindspore as ms
import mindspore.communication as D
D.init()
device_num = D.get_group_size()
rank_id = D.get_rank()
net = Net()
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
if rank_id == 0:
    ms.set_auto_parallel_context(strategy_ckpt_save_file="../dst_strategy.ckpt")
opt = Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())
model = ms.Model(net, optimizer=opt)
model.build(1, dataset)
```

其中，

- `dataset`：MindData对象，需要提前构造好以给入`model.build`。

当目标网络是进行推理时，则将`model.build`更换为`model.infer_preict_layout`以执行编译。

```python
import mindspore as ms
import mindspore.communication as D
D.init()
device_num = D.get_group_size()
rank_id = D.get_rank()
net = Net()
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
if rank_id == 0:
    ms.set_auto_parallel_context(strategy_ckpt_save_file="../dst_strategy.ckpt")
opt = Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())
model = ms.Model(net, optimizer=opt)
model.infer_predict_layout(Tensor(np.ones(shape=data_shape)))
```

其中，

- `data_shape`：推理数据的Shape。

示例中执行4卡目标网络编译的脚本执行命令为：

```bash
bash run_compile_4p.sh ../output/wmt14.en_fr.txt
```

执行后，将会生成目标切分策略文件:

```text
dst_strategy.ckpt
```

#### 执行分布式Checkpoint转换

示例中，原始策略以8卡进行训练，模型并行为4，数据并行为2，并开启优化器并行，策略文件命名为`src_strategy.ckpt`；
目标策略以4卡进行训练，模型并行为4，数据并行为1，不开启优化器并行，策略文件命名为`dst_stategy.ckpt`。

分布式Checkpoint提供两个接口对Checkpoint进行转换。第一个接口`transform_checkpoints`，要求用户将所有的Checkpoint放置于一个目录，并且子目录必须以”rank_0、rank_1、rank_2、...“格式进行命名。
用户调用该接口直接对整个目录进行转换。该方式使用较为方便，但是转换需要的内存开销会略高一些。第二个接口`transform_checkpoint_by_rank`，用以获取到特定的rank的Checkpoint，有更大的灵活性与更低的内存开销，
需要配合`rank_list_for_transform`接口使用，以获取本rank的目标Checkpoint需要哪些原始Checkpoint。

1. 使用接口`transform_checkpoints`。

   ```python
   import mindspore as ms
   ms.transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir,
                            "transformed", src_strategy_file, dst_strategy_file)
   ```

    > src_checkpoints_dir内的子目录要求按照”rank_x/checkpoint_x.ckpt“格式进行存储。

    示例中，对整个Checkpoint目录进行转换的脚本执行命令为：

    ```bash
    python transform_checkpoint_dir.py --src_strategy_file=./src_strategy.ckpt --dst_strategy_file=./dst_strategy.ckpt --src_checkpoints_dir=./src_checkpoints --dst_checkpoints_dir=./dst_checkpoints
    ```

2. 调用`transform_checkpoint_by_rank`接口对"transform_rank"进行参数合并。

   ```python
   import mindspore as ms
   rank_list = ms.rank_list_for_transform(transform_rank, src_strategy_file, dst_strategy_file)
   checkpoint_file_map = {}
   for rank_id in rank_list:
       checkpoint_file_map[rank_id] = os.path.join(src_checkpoints_dir, "rank_{}".format(rank_id), "src_checkpoint{}.ckpt".format(rank_id))
   save_checkpoint_path = os.path.join(src_checkpoints_dir, "rank_{}".format(transform_rank),
                                       "dst_checkpoint{}.ckpt".format(transform_rank))
   ms.transform_checkpoint_by_rank(transform_rank, checkpoint_file_map, save_checkpoint_path,
                                   src_strategy_file, dst_strategy_file)
   ```

    示例中，对Checkpoint按照rank逐个转换的脚本执行命令为：

    ```bash
    bash transform_by_rank.sh ./src_strategy.ckpt ./dst_strategy.ckpt ./src_checkpoints ./dst_checkpoints
    ```

执行后，将会生成转换后的目标Checkpoint文件目录:

```text
dst_checkpoints/
```

## 加载转换得到的Checkpoint文件

### 整体流程

对目标策略的网络进行编译，调用`load_checkpoint`接口，从转换后的Checkpoint文件中加载模型参数数据。

### 编译与执行目标网络

使用`model.build`(训练时)或者`model.infer_predict_layout`(推理时)接口对网络进行编译，此时权重Shape在编译流程进行了切分；调用`load_checkpoint`接口，从Checkpoint文件中加载各卡的模型参数数据。

目标网络是训练场景：

```python
import mindspore as ms
import mindspore.communication as D
D.init()
device_num = D.get_group_size()
rank_id = D.get_rank()
net = Net()
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
if rank_id == 0:
    ms.set_auto_parallel_context(strategy_ckpt_save_file="../dst_strategy.ckpt")
opt = Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())
model = ms.Model(net, optimizer=opt)
param_dict = ms.load_checkpoint(ckpt_file)
model.build(train_dataset=dataset, epoch=2)
ms.load_param_into_net(net, param_dict)
model.train(2, dataset, callbacks=callback, dataset_sink_mode=True)
```

- `ckpt_file`：需要加载的Checkpoint模型参数文件名称。

目标网络是推理场景：

```python
import mindspore as ms
import mindspore.communication as D
D.init()
device_num = D.get_group_size()
rank_id = D.get_rank()
net = Net()
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
if rank_id == 0:
    ms.set_auto_parallel_context(strategy_ckpt_save_file="../dst_strategy.ckpt")
opt = Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())
model = ms.Model(net, optimizer=opt)
param_dict = ms.load_checkpoint(ckpt_file)
model.infer_predict_layout(predict_data)
ms.load_param_into_net(net, param_dict)
model.predict(2, predict_data)
```

- `predict_data`：用于推理的Tensor数据。

示例中，加载转换后的Checkpoint进行二阶段微调训练的脚本执行命令为：

```bash
bash run_train_4p.sh ../output/wmt14.en_fr.txt
```

执行完成后，可以看到loss从6.45开始下降：

```text
epoch: 1 step: 73, loss is 6.45995
epoch: 1 step: 73, loss is 6.13733
```

## 流水线并行维度转换

[流水线并行](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/pipeline_parallel.html) 是对线性的网络进行切分，得到多个子网络，子网络之间在多卡间进行流水，因此每个子图存储下来的切分策略文件是不一致的，所有切分策略汇聚在一起才能得到完整的网络的切分信息。
因此针对流水线并行的维度，相比于其它维度的转换，需要事先执行一次汇聚切分策略文件的操作，得到汇聚后的切分策略文件，以这一份文件作为分布式Checkpoint转换依赖的策略文件。此外，与前一个章节[切分策略转换](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/resilience_train_and_predict.html#%E5%AF%B9%E5%88%86%E5%B8%83%E5%BC%8Fcheckpoint%E8%BF%9B%E8%A1%8C%E8%BD%AC%E6%8D%A2) 没有差异。

首先，执行8卡的流水线并行训练，其中pipeline并行维度为2，算子级模型并行维度为4，数据并行维度为1。

```python
import mindspore as ms
import mindspore.communication as D
D.init()
device_num = D.get_group_size()
rank_id = D.get_rank()
net = Net()
net = PipelineCell(net, 4) # micro_batch=4
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, pipeline_stages=2)
ms.set_auto_parallel_context(strategy_ckpt_save_file="../src_pipeline_strategys/src_strategy{}.ckpt")
opt = Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())
model = ms.Model(net, optimizer=opt)
ckpt_config = ms.CheckpointConfig(save_checkpoint_steps=callback_size, keep_checkpoint_max=1,
                                  integrated_save=False)
ckpoint_cb = ms.ModelCheckpoint(prefix="src_checkpoint",
                                directory = "../src_checkpoints/rank_{}".format(rank_id),
                                config=ckpt_config)
callback = [ms.TimeMonitor(callback_size), ms.LossMonitor(callback_size), ckpoint_cb]
model.train(2, dataset, callbacks=callback, dataset_sink_mode=True)
```

其中，

- `dataset`：MindData对象，需要提前构造好以给入`model.train`。

示例里面执行8卡训练脚本执行命令为：

```bash
bash run_train_8p_pipeline.sh ../output/wmt14.en_fr.txt
```

执行后，将会生成源Checkpoint文件目录以及源切分策略文件:

```text
src_checkpoints_pipeline/
src_pipeline_strategys/
```

参考[切分策略转换](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/resilience_train_and_predict.html#%E5%AF%B9%E5%88%86%E5%B8%83%E5%BC%8Fcheckpoint%E8%BF%9B%E8%A1%8C%E8%BD%AC%E6%8D%A2) 章节的“对目标网络执行编译”模块，同样编译目标网络以得到目标网络的切分策略文件。

示例中执行4卡目标网络编译的脚本执行命令为：

```bash
bash run_compile_4p.sh ../output/wmt14.en_fr.txt
```

执行后，将会生成目标切分策略文件:

```text
dst_strategy.ckpt
```

下一步展开包含pipeline并行维度的分布式Checkpoint维度转换，首先使用接口`merge_pipeline_strategys`对pipline训练得到的切分策略文件进行合并，而后使用接口`transform_checkpoints`或者`transform_checkpoint_by_rank`进行分布式Checkpoint转换。

示例给出使用`transform_checkpoints`的接口，使用`transform_checkpoint_by_rank`接口请参考[切分策略转换](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/resilience_train_and_predict.html#%E5%AF%B9%E5%88%86%E5%B8%83%E5%BC%8Fcheckpoint%E8%BF%9B%E8%A1%8C%E8%BD%AC%E6%8D%A2) 章节的介绍。

```python
import mindspore as ms
ms.merge_pipeline_strategys(src_pipeline_strategys_dir, src_strategy_file)
ms.transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir,
                            "transformed", src_strategy_file, dst_strategy_file)
```

> src_checkpoints_dir内的子目录要求按照"rank_x/checkpoint_x.ckpt"格式进行存储。

示例中，对整个Checkpoint目录进行转换的脚本执行命令为：

```bash
python transform_checkpoint_dir_pipeline.py --src_strategy_dir=./src_pipeline_strategys --dst_strategy_file=dst_strategy.ckpt --src_checkpoints_dir=./src_checkpoints --dst_checkpoints_dir=./dst_checkpoints
```

转换完成后，参照[执行目标网络章节](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/resilience_train_and_predict.html#%E5%8A%A0%E8%BD%BD%E8%BD%AC%E6%8D%A2%E5%BE%97%E5%88%B0%E7%9A%84checkpoint%E6%96%87%E4%BB%B6) ，加载转换得到的分布式Checkpoint，执行没有pipeline维度的分布式网络。

示例中，加载转换后的Checkpoint进行二阶段微调训练的脚本执行命令为：

```bash
bash run_train_4p.sh ../output/wmt14.en_fr.txt
```

执行完成后，可以看到loss从6.45开始下降：

```text
epoch: 1 step: 73, loss is 6.45995
epoch: 1 step: 73, loss is 6.13733
```
