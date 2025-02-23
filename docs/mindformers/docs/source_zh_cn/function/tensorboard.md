# TensorBoard可视化训练监控

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/function/tensorboard.md)

MindSpore Transformers 支持 TensorBoard 作为可视化工具，用于监控和分析训练过程中的各种指标和信息。TensorBoard 是一个独立的可视化库，需要用户手动安装，它提供了一种交互式的方式来查看训练中的损失、精度、学习率、梯度分布等多种内容。用户在训练`yaml`文件中配置 TensorBoard 后，在大模型训练过程中会实时生成并更新事件文件，通过 `tensorboard --logdir=/path/to/events.*` 命令查看训练数据。

## 配置说明

在训练`yaml`文件中进行如下配置，训练结束后会在配置的保存地址中保存事件文件：

### `yaml`文件配置样例

```yaml
seed: 0
output_dir: './output'

tensorboard:
    tensorboard_dir: 'worker/tensorboard'
    tensorboard_queue_size: 10
    log_loss_scale_to_tensorboard: True
    log_timers_to_tensorboard: True
```

| 参数名称                                      | 说明                                                      | 类型   |
|-------------------------------------------|---------------------------------------------------------|------|
| tensorboard.tensorboard_dir               | 设置 TensorBoard 事件文件的保存路径                                | str  |
| tensorboard.tensorboard_queue_size        | 设置采集队列的最大缓存值，超过该值便会写入事件文件，默认值为10                        | int  |
| tensorboard.log_loss_scale_to_tensorboard | 设置是否将 loss scale 信息记录到事件文件，默认为`False`                   | bool |
| tensorboard.log_timers_to_tensorboard     | 设置是否将计时器信息记录到事件文件，计时器信息包含当前训练步骤（或迭代）的时长以及吞吐量，默认为`False` | bool |

## 查看训练数据

进行上述配置后，训练期间将会在路径 `./worker/tensorboard/rank_{id}` 下保存每张卡的事件文件，其中 `{id}` 为实际的rank数。事件文件以 `events.*` 命名。文件中包含 `scalars` 和 `text` 数据，其中 `scalars` 为训练过程中关键指标的标量，如学习率、损失等； `text` 为训练任务所有配置的文本数据，如并行配置、数据集配置等。

使用以下命令可以启动 Tensorboard Web 可视化服务：

```bash
tensorboard --logdir=/path/to/events.* --host=0.0.0.0 --port=6006
```

 参数名称   | 说明                                                     |
|--------|--------------------------------------------------------|
| logdir | TensorBoard保存事件文件的文件夹路径                                |
| host   | 默认是 127.0.0.1，表示只允许本机访问；设置为 0.0.0.0 可以允许外部设备访问，请注意信息安全 |
| port   | 设置服务监听的端口，默认是 6006                                               |

输入样例中的命令后会显示：

```shell
TensorBoard 2.18.0 at http://0.0.0.0:6006/ (Press CTRL+C to quit)
```

其中 `2.18.0` 表示 TensorBoard 当前安装的版本号（推荐版本为 `2.18.0` ）， `0.0.0.0` 和 `6006` 分别对应输入的 `--host` 和 `--port` ，之后可以在本地PC的浏览器中访问 `服务器公共ip:端口号` 查看可视化页面，例如服务器的公共IP为 `192.168.1.1` ，则访问 `192.168.1.1:6006` 。

### 标量可视化说明

在 SCALARS 页面中，每个标量（假设名为 `scalar_name`）都存在 `scalar_name` 和 `scalar_name-vs-samples` 两个下拉标签页。其中 `scalar_name` 下展示了该标量随训练迭代步数进行变化的折线图； `scalar_name-vs-samples` 下展示了该标量随样本数进行变化的折线图。如下图所示：

![/tensorboard_scalar](./image/tensorboard_scalar.png)

所有标量的名称和说明如下：

| 标量名         | 说明                                                  |
|----------------|-----------------------------------------------------|
| learning-rate  | 学习率                                                 |
| batch-size     | 批次大小                                                |
| loss           | 损失                                                  |
| loss-scale     | 损失缩放因子，记录需要设置`log_loss_scale_to_tensorboard`为`True` |
| grad-norm      | 梯度范数                                                |
| iteration-time | 训练迭代所需的时间，记录需要设置`log_timers_to_tensorboard`为`True`  |
| through-put    | 吞吐量，记录需要设置`log_timers_to_tensorboard`为`True`        |

### 文本数据可视化说明

在 TEXT 页面中，每个训练配置存在一个标签页，其中记录了该配置的值。如下图所示：

![/tensorboard_text](./image/tensorboard_text.png)

所有配置名和说明如下：

| 配置名                        | 说明                                                           |
|----------------------------|--------------------------------------------------------------|
| seed                       | 随机种子                                                         |
| output_dir                 | 保存checkpoint、strategy的路径                                     |
| run_mode                   | 运行模式                                                         |
| use_parallel               | 是否开启并行                                                       |
| resume_training            | 是否开启断点续训功能                                                   |
| ignore_data_skip           | 是否忽略断点续训时跳过数据的机制，而从头开始读取数据集。只在 `resume_training` 值为`True`时记录 |
| data_skip_steps            | 数据集跳过步数。只在 `ignore_data_skip` 被记录且值为`False`时记录               |
| load_checkpoint            | 加载权重的模型名或权重路径                                                |
| load_ckpt_format           | 加载权重的文件格式。只在 `load_checkpoint` 值不为空时记录                       |
| auto_trans_ckpt            | 是否开启自动在线权重切分或转换。只在 `load_checkpoint` 值不为空时记录                 |
| transform_process_num      | 转换checkpoint的进程数。只在 `auto_trans_ckpt` 被记录且值为`True`时记录        |
| src_strategy_path_or_dir   | 源权重分布式策略文件路径。只在 `auto_trans_ckpt` 被记录且值为`True`时记录            |
| load_ckpt_async            | 是否异步记载权重。只在 `load_checkpoint` 值不为空时记录                        |
| only_save_strategy         | 任务是否仅保存分布式策略文件                                               |
| profile                    | 是否开启性能分析工具                                                   |
| profile_communication      | 是否在多设备训练中收集通信性能数据。只在 `profile` 值为`True`时记录                   |
| profile_level              | 采集性能数据级别。只在 `profile` 值为`True`时记录                            |
| profile_memory             | 是否收集Tensor内存数据。只在 `profile` 值为`True`时记录                      |
| profile_start_step         | 性能分析开始的step。只在 `profile` 值为`True`时记录                         |
| profile_stop_step          | 性能分析结束的step。只在 `profile` 值为`True`时记录                         |
| profile_rank_ids           | 指定rank ids开启profiling。只在 `profile` 值为`True`时记录               |
| profile_pipeline           | 是否按流水线并行每个stage的其中一张卡开启profiling。只在 `profile` 值为`True`时记录    |
| init_start_profile         | 是否在Profiler初始化的时候开启数据采集                                      |
| layer_decay                | 层衰减系数                                                        |
| layer_scale                | 是否启用层衰减                                                      |
| lr_scale                   | 是否开启学习率缩放                                                    |
| lr_scale_factor            | 学习率缩放系数。只在 `lr_scale` 值为`True`时记录                            |
| micro_batch_interleave_num | batch_size的拆分份数，多副本并行开关                                      |
| remote_save_url            | 使用AICC训练作业时，目标桶的回传文件夹路径                                      |
| callbacks                  | 回调函数配置                                                       |
| context                    | 环境配置                                                         |
| data_size                  | 数据集长度                                                        |
| device_num                 | 设备数量（卡数）                                                     |
| do_eval                    | 是否开启边训练边评估                                                   |
| eval_callbacks             | 评估回调函数配置。只在 `do_eval` 值为`True`时记录                            |
| eval_step_interval         | 评估step间隔。只在 `do_eval` 值为`True`时记录                            |
| eval_epoch_interval        | 评估epoch间隔。只在 `do_eval` 值为`True`时记录                           |
| eval_dataset               | 评估数据集配置。只在 `do_eval` 值为`True`时记录                             |
| eval_dataset_task          | 评估任务配置。只在 `do_eval` 值为`True`时记录                              |
| lr_schedule                | 学习率                                                          |
| metric                     | 评估函数                                                         |
| model                      | 模型配置                                                         |
| moe_config                 | 混合专家配置                                                       |
| optimizer                  | 优化器                                                          |
| parallel_config            | 并行策略配置                                                       |
| parallel                   | 自动并行配置                                                       |
| recompute_config           | 重计算配置                                                        |
| remove_redundancy          | checkpoint保存时是否去除冗余                                          |
| runner_config              | 运行配置                                                         |
| runner_wrapper             | wrapper配置                                                    |
| tensorboard                | TensorBoard配置                                                |
| train_dataset_task         | 训练任务配置                                                       |
| train_dataset              | 训练数据集配置                                                      |
| trainer                    | 训练流程配置                                                       |

> 上述训练配置来源于:
>
> 1. 用户在训练启动命令 `run_mindformer.py` 中传入的配置参数；
> 2. 用户在训练配置文件 `yaml` 中设置的配置参数；
> 3. 训练默认的配置参数。
>
> 可配置的所有参数请参考[配置文件说明](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/appendix/conf_files.html)。