# 其他特性

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/other_features.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## [切分策略传播](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/sharding_propagation.html)

在算子级并行中，需要用户配置正向网络中每个算子的切分策略（若不配置，则默认使用数据并行的策略）。而切分策略传播特性可以仅配置若干个算子的切分策略，为未配置切分策略的算子自动生成可行的切分策略，并且达到最小化通信开销的效果。

## [Parameter Server模式](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/parameter_server_training.html)

Parameter Server(参数服务器)是分布式训练中一种广泛使用的架构，相较于数据并行同步的AllReduce训练方法，Parameter Server具有更好的灵活性、可扩展性以及节点容灾的能力。参数服务器既支持同步SGD(Stochastic Gradient Descent，随机梯度下降)，也支持异步SGD的训练算法。在扩展性上，将模型的计算与模型的更新分别部署在Worker和Server两类进程中，使得Worker和Server的资源可以独立地横向扩缩(新增或者删除Worker和Server资源)。另外，在大规模数据中心的环境下，计算设备、网络以及存储经常会出现各种故障而导致部分节点异常，而在参数服务器的架构下，能够较为容易地处理此类故障而不会对训练中的任务产生影响。

## [通信算子融合](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/comm_fusion.html)

在分布式训练场景下，跨设备甚至跨节点的数据传输是制约扩展性以及算力利用率的瓶颈。通信算子融合是一种提升网络资源利用率、加速数据传输效率的重要方法，其将相同源节点和目的节点的通信算子打包同时执行，以避免多个单算子执行带来的额外开销。

## [数据集切分](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/dataset_slice.html)

在进行分布式训练时，需要将训练数据集导入到每个设备上。常见的导入方式有两种：1）以数据并行的方式导入，即将数据按batch维度进行切分，每个设备导入一部分。2）每个设备导入全量的数据。另外，当数据的某些维度特别大时（如遥感图片的H/W维可能特别大），即使样本数很少，也需要对图片进行切分，即将数据按H/W维度进行切分，每张设备读取一部分图片。此特性能支持将数据集按特定维度切分，以满足大幅面图片处理领域的训练诉求。

## [函数式算子切分](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/pynative_shard_function_parallel.html)

在动态图模式下，指定网络结构中的某个部分以图模式执行，并进行各种并行操作。

## [在K8s集群中使用ms-operator进行分布式训练](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/ms_operator.html)

MindSpore Operator 是MindSpore在Kubernetes上进行分布式训练的插件。CRD（Custom Resource Definition）中定义了Scheduler、PS、Worker三种角色，用户只需配置yaml文件，即可轻松实现分布式训练。

当前ms-operator支持普通单Worker训练、PS模式的单Worker训练以及自动并行（例如数据并行、模型并行等）的Scheduler、Worker启动。详细流程请参考[ms-operator](https://gitee.com/mindspore/ms-operator)。

## 特性相关接口说明

| 特性类别 &emsp; | 特性接口 | 说明 | 作用 |
| -------- | :------- | ---- | ---- |
| 自动并行 | set_auto_parallel_context(search_mode=mode) | 用于指定策略搜索算法，其值为字符串类型，可选值为：<br />1，"sharding_propagation"：表示使用切分策略传播算法进行策略搜索；<br />2，"dynamic_programming"：表示使用动态规划算法进行策略搜索；<br />3，"recursive_programming"：表示使用双递归算法进行策略搜索； | 自动并行可以让用户不配置或者少量配置算子的切分策略，而由框架搜索出切分策略。 |
|  | set_algo_parameters(fully_use_devices=bool_value) | 用于设置搜索策略时是否需要将算子切分到所有设备上。其值为bool类型，默认为True。 | 如果将算子切分到所有设备上，则能缩小搜索空间，提高搜索速度，但搜索出来的策略并非全局最优。 |
|  | set_auto_parallel_context(all_reduce_fusion_config=config) | 配置梯度AllReduce算子融合策略，其值为list类型。例如：[20, 35]，表示将前20个AllReduce融合成1个，第20～35个AllReduce融合成1个，剩下的AllReduce融合成1个。 | 减少AllReduce通信算子的操作次数，提高通信效率。 |
| 通信算子融合 | set_auto_parallel_context(comm_fusion=config) | 设置通信算子的融合配置，当前支持AllReduce、AllGather、ReduceScatter通信算子的配置。其值为dict类型，如comm_fusion={"allreduce": {"mode": "auto", "config": None}}。其中"mode"有三种选项：<br />"auto"：自动按照数据量阈值64MB进行算子融合，配置参数“config”为None。<br />"size"：按照手动设置数据量阈值的方式进行通信算子融合，配置参数"config"类型为int，单位MB。<br />"index"：仅"allreduce"支持配置index，表示按照通信算子序列号进行融合的方式，配置参数"config"类型为list。例如：[20, 35]，表示将前20个AllReduce融合成1个，第20～35个AllReduce融合成1个，剩下的AllReduce融合成1个。 | 减少AllReduce/AllGather/ReduceScatter通信算子的操作次数，提高通信效率。 |
| 数据集切分 | set_auto_parallel_context(dataset_strategy=config) | 配置数据集的切分策略。其中，config为Union[str, tuple]。<br />当传入字符串时，有两种选项：<br />"full_batch"：表示数据集不切分；<br />"data_parallel"：表示数据集按数据并行的方式切分。<br />当传入tuple时，tuple中的内容代表数据集的切分策略，类似于primitive的shard()接口。<br />若不调用此接口，则默认采用"data_parallel"的方式。 | 当样本数比卡数少时，可以采用"full_batch"的方式进行导入；当样本数大、模型参数小时，可以采用"data_parallel"的方式导入；当数据集是高分辨率图像数据时，可以采用配置tuple切分策略的方式导入。 |
| 分布式推理 | infer_predict_layout(*predict_data) | 使用推理数据进行一次预编译，输出算子的切分信息。 | 获取推理时所有权重的切分信息。 |
|  | load_distributed_checkpoint(network, checkpoint_filenames, predict_strategy=None, train_strategy_filename=None) | 加载分布式权重，需每台机器预先放置全量的ckpt。<br />其中network代表推理网络，checkpoint_filenames代表checkpoint文件，predict_strategy为infer_predict_layout()的输出，train_strategy_filename为训练时保存的算子切分策略信息。 | 加载分布式权重，以进行分布式推理。 |
| 函数式算子切分 | shard(in_strategy, out_strategy, device="Ascend", level=0)<br />在Cell类中 | 设置cell的输入及输出张量的切分策略，其余算子的并行策略由切分策略传播得到。 in_strategy/out_strategy指定输入/输出张量的切分策略，device指定执行设备，level指定切分策略传播算法的模式。 | 在PyNative模式下指定某个cell实例以图模式执行，并且依据指定的输入输出切分策略进行算子级别的模型并行， 其余的部分仍以PyNative模式执行数据并行。 |
|  | ops.shard(fn, in_strategy, out_strategy, device="Ascend", level=0) | 传入的fn为cell实例或函数，其余输入和shard相同，返回值为函数，再调用此函数时，会以图模式执行算子级别的模型并行。 | 此用法可以指定某个函数进行算子级别的模型并行，具体功能和cell的shard方法相同。|
