其他特性
========

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png
    :target: https://gitee.com/mindspore/docs/blob/master/tutorials/expert/source_zh_cn/parallel/other_features.rst

.. toctree::
  :maxdepth: 1
  :hidden:

  sharding_propagation
  parameter_server_training
  comm_fusion
  dataset_slice
  pynative_shard_function_parallel
  ms_operator

`切分策略传播 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/sharding_propagation.html>`__
-------------------------------------------------------------------------------------------------------------

在算子级并行中，需要用户配置正向网络中每个算子的切分策略（若不配置，则默认使用数据并行的策略）。而切分策略传播特性可以仅配置若干个算子的切分策略，为未配置切分策略的算子自动生成可行的切分策略，并且达到最小化通信开销的效果。

`Parameter Server模式 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/parameter_server_training.html>`__
--------------------------------------------------------------------------------------------------------------------------

Parameter
Server(参数服务器)是分布式训练中一种广泛使用的架构，相较于数据并行同步的AllReduce训练方法，Parameter
Server具有更好的灵活性、可扩展性以及节点容灾的能力。参数服务器既支持同步SGD(Stochastic
Gradient
Descent，随机梯度下降)，也支持异步SGD的训练算法。在扩展性上，将模型的计算与模型的更新分别部署在Worker和Server两类进程中，使得Worker和Server的资源可以独立地横向扩缩(新增或者删除Worker和Server资源)。另外，在大规模数据中心的环境下，计算设备、网络以及存储经常会出现各种故障而导致部分节点异常，而在参数服务器的架构下，能够较为容易地处理此类故障而不会对训练中的任务产生影响。

`通信算子融合 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/comm_fusion.html>`__
----------------------------------------------------------------------------------------------------

在分布式训练场景下，跨设备甚至跨节点的数据传输是制约扩展性以及算力利用率的瓶颈。通信算子融合是一种提升网络资源利用率、加速数据传输效率的重要方法，其将相同源节点和目的节点的通信算子打包同时执行，以避免多个单算子执行带来的额外开销。

`数据集切分 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/dataset_slice.html>`__
----------------------------------------------------------------------------------------------------

在进行分布式训练时，需要将训练数据集导入到每个设备上。常见的导入方式有两种：1）以数据并行的方式导入，即将数据按batch维度进行切分，每个设备导入一部分。2）每个设备导入全量的数据。另外，当数据的某些维度特别大时（如遥感图片的H/W维可能特别大），即使样本数很少，也需要对图片进行切分，即将数据按H/W维度进行切分，每张设备读取一部分图片。此特性能支持将数据集按特定维度切分，以满足大幅面图片处理领域的训练诉求。

`函数式算子切分 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/pynative_shard_function_parallel.html>`__
---------------------------------------------------------------------------------------------------------------------------

在动态图模式下，指定网络结构中的某个部分以图模式执行，并进行各种并行操作。

`在K8s集群中使用ms-operator进行分布式训练 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/ms_operator.html>`__
--------------------------------------------------------------------------------------------------------------------------------

MindSpore Operator
是MindSpore在Kubernetes上进行分布式训练的插件。CRD（Custom Resource
Definition）中定义了Scheduler、PS、Worker三种角色，用户只需配置yaml文件，即可轻松实现分布式训练。

当前ms-operator支持普通单Worker训练、PS模式的单Worker训练以及自动并行（例如数据并行、模型并行等）的Scheduler、Worker启动。详细流程请参考\ `ms-operator <https://gitee.com/mindspore/ms-operator>`__\ 。

特性相关接口说明
----------------

+-----------------------+-----------------------+--------------------+--------------------+
| 特性类别              | 特性接口              | 说明               | 作用               |
+=======================+=======================+====================+====================+
| 自动并行              | set_auto_parallel_con | 用于指定策略搜索算 | 自动并行可以让用户 |
|                       | text(search_mode=mode | 法，其值为字符串类 | 不配置或者少量配置 |
|                       | )                     | 型，可选值为：1，  | 算子的切分策略，而 |
|                       |                       | “sharding          | 由框架搜索出切分策 |
|                       |                       | _propagat          | 略。               |
|                       |                       | ion”：表示使用     |                    |
|                       |                       | 切分策略传播算法进 |                    |
|                       |                       | 行策略搜索；2，“   |                    |
|                       |                       | dynamic_p          |                    |
|                       |                       | rogrammin          |                    |
|                       |                       | g”：表示使用动态   |                    |
|                       |                       | 规划算法进行策略搜 |                    |
|                       |                       | 索；3，“recu       |                    |
|                       |                       | rsive_pro          |                    |
|                       |                       | gramming”          |                    |
|                       |                       | ：表示使用双递归算 |                    |
|                       |                       | 法进行策略搜索；   |                    |
+-----------------------+-----------------------+--------------------+--------------------+
|                       | set_algo_parameters(f | 用于设置搜索策略时 | 如果将算子切分到所 |
|                       | ully_use_devices=bool | 是否需要将算子切分 | 有设备上，则能缩小 |
|                       | _value)               | 到所有设备上。其值 | 搜索空间，提高搜索 |
|                       |                       | 为bool类型，默     | 速度，但搜索出来的 |
|                       |                       | 认为True。         | 策略并非全局最优。 |
+-----------------------+-----------------------+--------------------+--------------------+
|                       | set_auto_parallel_con | 配置梯度AllRe      | 减少AllRedu        |
|                       | text(all_reduce_fusio | duce算子融合策     | ce通信算子的操作   |
|                       | n_config=config)      | 略，其值为list     | 次数，提高通信效率 |
|                       |                       | 类型。例如：[20    | 。                 |
|                       |                       | ,                  |                    |
|                       |                       | 35]，表示将前2     |                    |
|                       |                       | 0个AllRedu         |                    |
|                       |                       | ce融合成1个，第    |                    |
|                       |                       | 20～35个All        |                    |
|                       |                       | Reduce融合成       |                    |
|                       |                       | 1个，剩下的All     |                    |
|                       |                       | Reduce融合成       |                    |
|                       |                       | 1个。              |                    |
+-----------------------+-----------------------+--------------------+--------------------+
| 通信算子融合          | set_auto_parallel_con | 设置通信算子的融合 | 减少AllRedu        |
|                       | text(comm_fusion=conf | 配置，当前支持Al   | ce/AllGat          |
|                       | ig)                   | lReduce、A         | her/Reduc          |
|                       |                       | llGather、         | eScatter通         |
|                       |                       | ReduceSca          | 信算子的操作次数， |
|                       |                       | tter通信算子的     | 提高通信效率。     |
|                       |                       | 配置。其值为dic    |                    |
|                       |                       | t类型，如comm      |                    |
|                       |                       | _fusion={          |                    |
|                       |                       | “allreduc          |                    |
|                       |                       | e”:                |                    |
|                       |                       | {“mode”:           |                    |
|                       |                       | “auto”,            |                    |
|                       |                       | “config”:          |                    |
|                       |                       | None}}。其中       |                    |
|                       |                       | “mode”有三种       |                    |
|                       |                       | 选项：“auto”       |                    |
|                       |                       | ：自动按照数据量阈 |                    |
|                       |                       | 值64MB进行算子     |                    |
|                       |                       | 融合，配置参数“c   |                    |
|                       |                       | onfig”为No         |                    |
|                       |                       | ne。“size”         |                    |
|                       |                       | ：按照手动设置数据 |                    |
|                       |                       | 量阈值的方式进行通 |                    |
|                       |                       | 信算子融合，配置参 |                    |
|                       |                       | 数“config”         |                    |
|                       |                       | 类型为int，单位    |                    |
|                       |                       | MB。“index         |                    |
|                       |                       | ”：仅“allre        |                    |
|                       |                       | duce”支持配置      |                    |
|                       |                       | index，表示按      |                    |
|                       |                       | 照通信算子序列号进 |                    |
|                       |                       | 行融合的方式，配置 |                    |
|                       |                       | 参数“config        |                    |
|                       |                       | ”类型为list。      |                    |
|                       |                       | 例如：[20,         |                    |
|                       |                       |                    |                    |
|                       |                       | 35]，表示将前2     |                    |
|                       |                       | 0个AllRedu         |                    |
|                       |                       | ce融合成1个，第    |                    |
|                       |                       | 20～35个All        |                    |
|                       |                       | Reduce融合成       |                    |
|                       |                       | 1个，剩下的All     |                    |
|                       |                       | Reduce融合成       |                    |
|                       |                       | 1个。              |                    |
+-----------------------+-----------------------+--------------------+--------------------+
| 数据集切分            | set_auto_parallel_con | 配置数据集的切分策 | 当样本数比卡数少时 |
|                       | text(dataset_strategy | 略。其中，conf     | ，可以采用         |
|                       | =config)              | ig为Union[         | “full_batch”的     |
|                       |                       | str,               | 方式进行导入；当样 |
|                       |                       | tuple]。当传       | 本数大、模型参数小 |
|                       |                       | 入字符串时，有两种 | 时，可以采用“da    |
|                       |                       | 选项：“full_batch” | ta_parall          |
|                       |                       | ：表示             | el”的方式导入；    |
|                       |                       | 数据集不切分；“d   | 当数据集是高分辨率 |
|                       |                       | ata_paral          | 图像数据时，可以采 |
|                       |                       | lel”：表示数据     | 用配置tuple切      |
|                       |                       | 集按数据并行的方式 | 分策略的方式导入。 |
|                       |                       | 切分。当传入tup    |                    |
|                       |                       | le时，tuple        |                    |
|                       |                       | 中的内容代表数据集 |                    |
|                       |                       | 的切分策略，类似于 |                    |
|                       |                       | primitive          |                    |
|                       |                       | 的shard()接        |                    |
|                       |                       | 口。若不调用此接口 |                    |
|                       |                       | ，则默认采用“da    |                    |
|                       |                       | ta_parall          |                    |
|                       |                       | el”的方式。        |                    |
+-----------------------+-----------------------+--------------------+--------------------+
| 分布式推理            | infer_predict_layout( | 使用推理数据进行一 | 获取推理时所有权重 |
|                       | \*predict_data)       | 次预编译，输出算子 | 的切分信息。       |
|                       |                       | 的切分信息。       |                    |
+-----------------------+-----------------------+--------------------+--------------------+
|                       | load_distributed_chec | 加载分布式权重，需 | 加载分布式权重，以 |
|                       | kpoint(network,       | 每台机器预先放置全 | 进行分布式推理。   |
|                       | checkpoint_filenames, | 量的ckpt。其中     |                    |
|                       | predict_strategy=None | network代表        |                    |
|                       | ,                     | 推理网络，chec     |                    |
|                       | train_strategy_filena | kpoint_fi          |                    |
|                       | me=None)              | lenames代表        |                    |
|                       |                       | checkpoin          |                    |
|                       |                       | t文件，predi       |                    |
|                       |                       | ct_strategy        |                    |
|                       |                       | 为                 |                    |
|                       |                       | infer_predict_l    |                    |
|                       |                       | ayout()的输        |                    |
|                       |                       | 出，train_s        |                    |
|                       |                       | trategy_f          |                    |
|                       |                       | ilename为训        |                    |
|                       |                       | 练时保存的算子切分 |                    |
|                       |                       | 策略信息。         |                    |
+-----------------------+-----------------------+--------------------+--------------------+
| 函数式算子切分        | shard(in_strategy,    | 设置cell的输入     | 在PyNative         |
|                       | out_strategy,         | 及输出张量的切分策 | 模式下指定某个ce   |
|                       | device=“Ascend”,      | 略，其余算子的并行 | ll实例以图模式执   |
|                       | level=0)在Cell类中    | 策略由切分策略传播 | 行，并且依据指定的 |
|                       |                       | 得到。             | 输入输出切分策略进 |
|                       |                       | in_strate          | 行算子级别的模型并 |
|                       |                       | gy/out_st          | 行，               |
|                       |                       | rategy指定输       | 其余的部分仍以Py   |
|                       |                       | 入/输出张量的切分  | Native模式执       |
|                       |                       | 策略，device       | 行数据并行。       |
|                       |                       | 指定执行设备，le   |                    |
|                       |                       | vel指定切分策略    |                    |
|                       |                       | 传播算法的模式。   |                    |
+-----------------------+-----------------------+--------------------+--------------------+
|                       | ops.shard(fn,         | 传入的fn为cel      | 此用法可以指定某个 |
|                       | in_strategy,          | l实例或函数，其余  | 函数进行算子级别的 |
|                       | out_strategy,         | 输入和shard相      | 模型并行，具体功能 |
|                       | device=“Ascend”,      | 同，返回值为函数， | 和cell的sha        |
|                       | level=0)              | 再调用此函数时，会 | rd方法相同。       |
|                       |                       | 以图模式执行算子级 |                    |
|                       |                       | 别的模型并行。     |                    |
+-----------------------+-----------------------+--------------------+--------------------+
