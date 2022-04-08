# 分布式并行高级特性概述

`Ascend` `GPU` `分布式并行`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/distributed_advanced_overview.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 背景

随着深度学习的发展，模型规模越来越大。如NLP领域，短短几年时间，参数量就从BERT的亿级，发展到GPT-3的1700亿，再到盘古alpha 2000亿，以及当前业界甚至提出百万亿级。由此可以看出，近年来参数规模呈指数增长趋势。另一方面，随着大数据、互联网等领域相关技术的发展，可供模型训练的数据集也极速扩增，例如推荐、自然语言处理等场景的数据集可达数TB。

面对大数据量、大规模参数的训练，单个设备要么完成模型训练的时间很长，要么因显存不足而导致无法进行训练。因此，需要引入分布式训练技术。

当前，最常用的分布式训练技术是数据并行。数据并行将训练数据切分到多个设备上，每个设备维护相同的模型参数和相同大小的计算任务，但是处理不同的数据，并在反向传播过程中，对每个设备产生的参数梯度进行全局AllReduce同步求和。当数据集较大而模型较小时，选择数据并行较有优势，如ResNet50。但是，当模型规模较大、或数据集与模型规模均较大时，就需要借助于其他分布式特性。

MindSpore提供以下高级特性来支撑大模型分布式训练，用户可以根据自己的需要进行灵活组合。

## 分布式并行高级特性

### 算子级并行

算子级并行是以算子为单位，对其输入张量切分到多个设备，从而将算子进行分布式计算。一方面，可以将数据样本及模型参数同时切分到多个设备上，以完成大模型的训练。另一方面，可以充分利用集群资源进行并行计算，以提高整体速度。

用户可以设置正向网络中每个算子的切分策略，框架根据算子的切分策略对每个算子及其输入张量进行切分建模，使得该算子的计算逻辑在切分前后保持数学等价。

### 流水线并行

当集群设备数很多时，如果仅采用算子级并行的方式，则需要在整个集群的通信域上进行通信，这可能使得通信效率低，从而降低整体性能。

而流水线并行能将神经网络结构切分成多个stage，每个stage跑在一部分设备内，将集合通信的通信域限定在这部分设备范围内，而stage间采用点对点通信。

流水线并行的优点在于：能提升通信效率、能方便的处理按层堆叠的神经网络结构。缺点在于：同一时刻内，有些节点可能处于空闲状态。

### 优化器并行

在数据并行或算子级并行训练时，模型的参数可能在多个设备上存在同一份副本。这使得优化器在更新该权重之时，在多个设备间存在冗余计算。在此情况下，可以通过优化器并行将优化器的计算量分散到多个设备上。它的优点在于：能减少静态内存消耗、减少优化器内的计算量。缺点在于：增加了通信开销。

### Host&Device异构

在大模型训练时，因每个设备（加速器）的内存容量有限，从而总体所能训练的模型规模将受设备数的限制。为了能完成更大规模的模型训练，可以使用主机端（Host）和加速器（Device）异构的训练模式。它同时发挥了主机端内存大和加速器端计算快的优势，是超大模型训练过程中减少设备数的有效方式。

### 重计算

MindSpore根据正向图计算流程来自动推导出反向图，正向图和反向图一起构成了完整的计算图。在计算某些反向算子时，可能需要用到某些正向算子的计算结果，导致这些正向算子的计算结果，需要驻留在内存中直到这些反向算子计算完，它们所占的内存才会被其他算子复用。而这些正向算子的计算结果，长时间驻留在内存中，会推高计算的内存占用峰值，在大规模网络模型中尤为显著。为了降低内存峰值，重计算技术可以不保存正向激活层的计算结果，让该内存可以被复用，然后在计算反向部分时，重新计算出正向激活层的结果。

### 切分策略传播

在算子级并行中，需要用户配置正向网络中每个算子的切分策略（若不配置，则默认使用数据并行的策略）。而切分策略传播特性可以仅配置若干个算子的切分策略，为未配置切分策略的算子自动生成可行的切分策略，并且达到最小化通信开销的效果。

### Parameter Server模式

Parameter Server(参数服务器)是分布式训练中一种广泛使用的架构，相较于数据并行同步的AllReduce训练方法，Parameter Server具有更好的灵活性、可扩展性以及节点容灾的能力。参数服务器既支持同步SGD(Stochastic Gradient Descent，随机梯度下降)，也支持异步SGD的训练算法。在扩展性上，将模型的计算与模型的更新分别部署在Worker和Server两类进程中，使得Worker和Server的资源可以独立地横向扩缩(新增或者删除Worker和Server资源)。另外，在大规模数据中心的环境下，计算设备、网络以及存储经常会出现各种故障而导致部分节点异常，而在参数服务器的架构下，能够较为容易地处理此类故障而不会对训练中的任务产生影响。

### 通信算子融合

在分布式训练场景下，跨设备甚至跨节点的数据传输是制约扩展性以及算力利用率的瓶颈。通信算子融合是一种提升网络资源利用率、加速数据传输效率的重要方法，其将相同源节点和目的节点的通信算子打包同时执行，以避免多个单算子执行带来的额外开销。

### 数据集切分

在进行分布式训练时，需要将训练数据集导入到每个设备上。常见的导入方式有两种：1）以数据并行的方式导入，即将数据按batch维度进行切分，每个设备导入一部分。2）每个设备导入全量的数据。另外，当数据的某些维度特别大时（如遥感图片的H/W维可能特别大），即使样本数很少，也需要对图片进行切分，即将数据按H/W维度进行切分，每张设备读取一部分图片。此特性能支持将数据集按特定维度切分，以满足大幅面图片处理领域的训练诉求。

### 分布式推理

对于大模型来说，训练好的参数巨大，如盘古alpha拥有2000亿参数。如此庞大的模型在推理时，无法简单地部署在单个设备上，推理时需要把模型切分到集群中。并且，推理和训练可能不在同一个环境中，所使用的集群大小可能不一样。因此，还需支持推理设备数与训练设备数不一致的场景。

### 函数式算子切分

在动态图模式下，指定网络结构中的某个部分以图模式执行，并进行各种并行操作。

## 特性相关接口说明

| 特性类别 | 特性接口 | 说明 | 作用 |
| -------- | :------- | ---- | ---- |
| 算子级并行 | shard(in_strategy=None, out_strategy=None)<br />在Primitive类中 | 设置算子的输入及输出张量的切分策略（其中，输出张量的切分策略仅支持部分算子，如Gather、MatMul） | 通过将网络模型中每个算子涉及到的张量进行切分，降低单个设备的内存容量，以完成大模型训练/推理。或利用集群资源，进行分布式计算，减少整体执行时间。 |
|          | add_prim_attr(name, value)<br />在Primitive类中 | Gather算子：<br />add_prim_attr(“manual_split”, config)：配置其第一个输入的非均匀切分策略，其中config类型为tuple，用于描述第一个参数第0维的切分方式。比如(10, 20, 30, 4)代表将算子第一个输入的第0维切分成4份，每份的shape大小分别为10，20，30，4。 | 在推荐领域，存在数据集的每一列对应一个子表的场景。在该场景下，使用此配置能降低通信量，提升整体性能。 |
| |  | EmbeddingLookUp算子：<br />add_prim_attr(“primitive_target”, “CPU”)：配置其在CPU上执行，用于异构场景。 | 在推荐领域，存在Embedding Table特别大的场景，为了节约device内存，可以使用此配置将EmbeddingLookUp放到CPU上执行，以完成推荐大模型的训练。 |
| | set_auto_parallel_context(enable_alltoall=bool_value) | 表示在通信时是否允许产生AllToAll通信算子，其值为bool类型，默认为False。 | AllToAll通信能减少通信数据量，提高通信效率，但需要环境支持。 |
| 流水线并行 | set_auto_parallel_context(pipeline_stages=stage_num) | 设置流水线并行的stage个数，其值为正整数，取值范围为[1, 设备数]。 | 指定stage的个数，将集合通信的通信域限定在stage范围内，而stage间采用点对点通信。 |
| | pipeline_stage(value)<br />在Cell类中 | 设置该Cell在哪个stage中执行。 | 设置该Cell在哪个stage中执行。 |
| | PipelineCell(network, micro_size) | 用于指定训练网络的MicroSize数量，其中network为待训练的网络，micro_size为正整数。 | 指定micro_size，能减少stage间的空闲等待时间，提升流水线并行的整体效率。 |
| 优化器并行 | set_auto_parallel_context(enable_parallel_optimizer=bool_value) | 表示是否开启优化器并行，其值为bool型，默认为False。 | 优化器并行能节省静态内存的开销，但增加了通信开销。 |
|  | set_auto_parallel_context(parallel_optimizer_config=config) | 只有开启优化器并行后，此配置才生效。其中config是个dict，支持两个键值：<br />gradient_accumulation_shard(bool)：如果为True，则累积梯度变量将在数据并行度上进行分片，默认为False。<br />parallel_optimizer_threshold(int)：该值表示优化器切分阈值，单位为KB（默认64KB）。当参数大小不超过该值时，将不会被切分。 | gradient_accumulation_shard为True时，将节省一份参数大小的静态内存，但增加了通信开销。<br />优化器切分阈值，能使得shape较小的参数不进行优化器切分，以节省通信资源。 |
| 重计算 | recompute(mode=True)<br />在primitive类中 | 用于指定该算子是否需要重计算，其值为bool类型，默认为True，表示开启算子重计算。 | 开启算子重计算后，能减少动态内存的峰值，但增加整体计算量。 |
|  | recompute(**kwargs)<br />在Cell类中 | 调用此接口后，将会对此Cell中的算子进行重计算。<br />其中输入参数有两个bool类型选项：<br />mp_comm_recompute：是否开启模型并行通信算子重计算，默认为True。<br />parallel_optimizer_comm_recompute：是否开启优化器并行通信算子重计算，默认为False | 开启Cell重计算，且能配置模型并行的通信算子、优化器并行的通信算子是否进行重计算。当通信算子重计算时，将消耗通信资源，但能降低动态内存的峰值。 |
| 自动并行 | set_auto_parallel_context(search_mode=mode) | 用于指定策略搜索算法，其值为字符串类型，可选值为：<br />1，"sharding_propagation"：表示使用切分策略传播算法进行策略搜索；<br />2，"dynamic_programming"：表示使用动态规划算法进行策略搜索；<br />3，"recursive_programming"：表示使用双递归算法进行策略搜索； | 自动并行可以让用户不配置或者少量配置算子的切分策略，而由框架搜索出切分策略。 |
|  | set_algo_parameters(fully_use_devices=bool_value) | 用于设置搜索策略时是否需要将算子切分到所有设备上。其值为bool类型，默认为True。 | 如果将算子切分到所有设备上，则能缩小搜索空间，提高搜索速度，但搜索出来的策略并非全局最优。 |
|  | set_auto_parallel_context(all_reduce_fusion_config=config) | 配置梯度AllReduce算子融合策略，其值为list类型。例如：[20, 35]，表示将前20个AllReduce融合成1个，第20～35个AllReduce融合成1个，剩下的AllReduce融合成1个。 | 减少AllReduce通信算子的操作次数，提高通信效率。 |
| 通信算子融合 | set_auto_parallel_context(comm_fusion=config) | 设置通信算子的融合配置，当前支持AllReduce、AllGather、ReduceScatter通信算子的配置。其值为dict类型，如comm_fusion={"allreduce": {"mode": "auto", "config": None}}。其中"mode"有三种选项：<br />"auto"：自动按照数据量阈值64MB进行算子融合，配置参数“config”为None。<br />"size"：按照手动设置数据量阈值的方式进行通信算子融合，配置参数"config"类型为int，单位MB。<br />"index"：仅"allreduce"支持配置index，表示按照通信算子序列号进行融合的方式，配置参数"config"类型为list。例如：[20, 35]，表示将前20个AllReduce融合成1个，第20～35个AllReduce融合成1个，剩下的AllReduce融合成1个。 | 减少AllReduce/AllGather/ReduceScatter通信算子的操作次数，提高通信效率 |
| 数据集切分 | set_auto_parallel_context(dataset_strategy=config) | 配置数据集的切分策略。其中，config为Union[str, tuple]。<br />当传入字符串时，有两种选项：<br />"full_batch"：表示数据集不切分；<br />"data_parallel"：表示数据集按数据并行的方式切分。<br />当传入tuple时，tuple中的内容代表数据集的切分策略，类似于primitive的shard()接口。<br />若不调用此接口，则默认采用"data_parallel"的方式。 | 当样本数比卡数少时，可以采用"full_batch"的方式进行导入；当样本数大、模型参数小时，可以采用"data_parallel"的方式导入；当数据集是高分辨率图像数据时，可以采用配置tuple切分策略的方式导入。 |
| 分布式推理 | infer_predict_layout(*predict_data) | 使用推理数据进行一次预编译，输出算子的切分信息。 | 获取推理时所有权重的切分信息。 |
|  | load_distributed_checkpoint(network, checkpoint_filenames, predict_strategy=None, train_strategy_filename=None) | 加载分布式权重，需每台机器预先放置全量的ckpt。<br />其中network代表推理网络，checkpoint_filenames代表checkpoint文件，predict_strategy为infer_predict_layout()的输出，train_strategy_filename为训练时保存的算子切分策略信息。 | 加载分布式权重，以进行分布式推理。 |
| 函数式算子切分 | shard(in_strategy, out_strategy, device="Ascend", level=0)<br />在cell类中 | 设置cell的输入及输出张量的切分策略，其余算子的并行策略由切分策略传播得到。 in_strategy/out_strategy指定输入/输出张量的切分策略，device指定执行设备，level指定切分策略传播算法的模式。 | 在PyNative模式下指定某个cell实例以图模式执行，并且依据指定的输入输出切分策略进行算子级别的模型并行， 其余的部分仍以PyNative模式执行数据并行。 |
|  | ops.shard(fn, in_strategy, out_strategy, device="Ascend", level=0) | 传入的fn为cell实例或函数，其余输入和shard相同，返回值为函数，再调用此函数时，会以图模式执行算子级别的模型并行 | 此用法可以指定某个函数进行算子级别的模型并行，具体功能和cell的shard方法相同。|
