环境变量
========

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg
    :target: https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/api_python/env_var_list.rst
    :alt: 查看源文件

本文介绍MindSpore的环境变量。

数据处理
---------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - 环境变量
     - 功能
     - 类型
     - 取值
     - 说明
   * - DATASET_ENABLE_NUMA
     - 是否开启Dataset模块的numa绑核功能，在大多数分布式场景下numa绑核都能提升数据处理效率和端到端性能
     - String
     - True: 开启Dataset模块的numa绑核功能
     - 与libnuma.so配合使用
   * - MS_CACHE_HOST
     - 开启cache时，cache服务所在的IP
     - String
     - Cache Server所在机器的IP
     - 与MS_CACHE_PORT一起使用
   * - MS_CACHE_PORT
     - 开启cache时，cache服务所在的端口
     - String
     - Cache Server所在机器的端口
     - 与MS_CACHE_HOST一起使用
   * - MS_DATASET_SINK_QUEUE
     - 指定数据下沉队列的容量大小
     - Integer
     - 1~128：有效的队列容量大小设置范围
     -
   * - MS_DEV_MINDRECORD_SHARD_BY_BLOCK
     - 当对MindRecord进行分片采样时，是否切换数据的切分策略为按块采样
     - String
     - True: 数据的切分策略为按块采样

       False: 数据的切分策略为按片采样
     - 默认值：False。只对DistributedSampler采样器生效。
   * - MS_ENABLE_NUMA
     - 是否开启全局numa绑核功能，提升端到端性能
     - String
     - True: 开启全局numa绑核功能
     -
   * - MS_FREE_DISK_CHECK
     - 是否开启剩余磁盘空间检查
     - String
     - True: 开启剩余磁盘空间检查

       False: 关闭剩余磁盘空间检查
     - 默认值：True，在使用多个并发同时在共享存储上创建MindRecord时，建议设置为False。
   * - MS_INDEPENDENT_DATASET
     - 是否开启Dataset独立进程模式，此时Dataset会运行于独立的子进程中，仅支持Linux平台
     - String
     - True: 开启Dataset独立进程模式

       False: 关闭Dataset独立进程模式
     - 默认值：False。此功能当前处于Beta测试阶段，不支持与AutoTune、Offload、Cache、DSCallback一同使用。如果在使用中遇到问题，欢迎反馈。
   * - OPTIMIZE
     - 是否执行dataset数据处理 pipeline 树优化，在适合数据处理算子融合的场景下，可以提升数据处理效率
     - String
     - true: 开启pipeline树优化

       false: 关闭pipeline树优化
     -

具体用法详见 `单节点数据缓存 <https://mindspore.cn/tutorials/zh-CN/master/dataset/cache.html>`_
和 `数据处理性能优化 <https://mindspore.cn/tutorials/zh-CN/master/dataset/optimize.html>`_ 。

图编译执行
----------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - 环境变量
     - 功能
     - 类型
     - 取值
     - 说明
   * - MS_DEV_JIT_SYNTAX_LEVEL
     - 指定静态图模式的语法支持级别
     - Integer
     - 0：指定静态图模式的语法支持级别为STRICT，仅支持基础语法，且执行性能最佳。可用于MindIR导入导出。

       2：指定静态图模式的语法支持级别为LAX，支持更多复杂语法，最大程度地兼容Python所有语法。由于存在可能无法导出的语法，不能用于MindIR导入导出。
     -
   * - MS_JIT_MODULES
     - 指定静态图模式下哪些模块需要JIT静态编译，其函数方法会被编译成静态计算图
     - String
     - 模块名，对应import导入的顶层模块的名称。如果有多个，使用英文逗号分隔。例如： `export MS_JIT_MODULES=mindflow,mindyolo`。
     - 默认情况下，第三方库之外的模块都会进行JIT静态编译。MindSpore套件等一些模块如 `mindflow`、`mindyolo` 等并不会被视作第三方库。如果有类似MindSpore套件的模块，内部存在 `nn.Cell`、`@ms.jit` 修饰函数或需要编译成静态计算图的函数方法，可以通过配置该环境变量，使该模块进行JIT静态编译而不会被当成第三方库。
   * - MS_JIT_IGNORE_MODULES
     - 指定静态图模式下哪些模块是第三方库，不进行JIT静态编译，其函数方法会被解释执行。
     - String
     - 模块名，对应import导入的顶层模块的名称。如果有多个，使用英文逗号分隔。例如： `export MS_JIT_IGNORE_MODULES=numpy,scipy`。
     - 静态图模式能够自动识别第三方库，一般情况下不需要为NumPy、SciPy这些可识别的第三方库设置该环境变量。如果 `MS_JIT_IGNORE_MODULES` 和 `MS_JIT_MODULES` 同时指定同一个模块名，前者生效，后者不生效。
   * - MS_DEV_FALLBACK_DUMP_NODE
     - 是否打印代码中由静态图语法增强技术支持的语法表达式
     - Integer
     - 1：开启打印功能。

       不设置或其他值：关闭打印功能。
     -
   * - MS_JIT
     - 是否使用JIT即时编译
     - Integer
     - 0：不使用JIT即时编译，网络脚本直接按照动态图（PyNative）模式执行。

       不设置或其他值：根据网络脚本判断执行静态图（Graph）模式还是动态图（PyNative）模式。
     -
   * - MS_DEV_FORCE_USE_COMPILE_CACHE
     - 是否直接使用编译缓存，不检查网络脚本有无被修改
     - Integer
     - 1：不检查网络脚本是否被修改，直接读取编译缓存。建议只在调试过程中使用，例如网络脚本只增加了print语句用于打印调试。

       不设置或其他值：检测网络脚本的改动，网络没有被修改时，才读取编译缓存。
     -
   * - MS_DEV_SIDE_EFFECT_LOAD_ELIM
     - 优化冗余显存拷贝操作
     - Integer
     - 0: 不做显存优化，占用显存最多。

       1: 保守地做部分显存优化。

       2: 在损耗一定编译性能的前提下，尽量多地优化显存。

       3: 不保证网络的精度，显存消耗最少。

       默认值：1
     -
   * - MS_DEV_SAVE_GRAPHS
     - 是否保存IR文件
     - Integer
     - 0：不保存IR文件。

       1：运行时会输出图编译过程中产生的一些中间文件。

       2：在等级1的基础上，生成更多后端流程相关的IR文件。

       3：在等级2的基础上，生成可视化计算图和更多详细的前端IR文件。
     -
   * - MS_DEV_SAVE_GRAPHS_PATH
     - 设置保存计算图的路径
     - String
     - 保存计算图的路径
     -
   * - MS_DEV_DUMP_IR_FORMAT
     - 配置IR图中展示哪些信息
     - Integer
     - 0：除return节点外，只打印节点的operator和节点的输入，并且简化子图的打印信息。

       1：打印除debug info和scope以外的所有信息。

       2或不设置：打印所有信息。
     -
   * - MS_DEV_DUMP_IR_INTERVAL
     - 设置间隔多少个IR文件打印保存一个IR文件，减少IR图的打印数量。
     - Integer
     - 1或不设置：打印保存所有IR文件。

       其他数值：按照指定的间隔个数保存IR文件。
     - 该环境变量与MS_DEV_DUMP_IR_PASSES同时打开时，优先遵从MS_DEV_DUMP_IR_PASSES的规则，该环境变量不会生效。
   * - MS_DEV_DUMP_IR_PASSES
     - 根据文件名指定保存哪些IR文件。
     - String
     - 文件名或文件名的一部分。如果有多个，使用逗号隔开。例如 `export MS_DEV_DUMP_IR_PASSES=recompute,renormalize`。
     - 设置该环境变量时，无论MS_DEV_SAVE_GRAPHS设置为什么等级，详细的前端IR文件都会参与筛选和打印。
   * - MS_DEV_DUMP_IR_PARALLEL_DETAIL
     - 控制是否打印 DUMP IR 图的详细信息 tensor_map 和 device_matrix。
     - Integer
     - 1: 打印 DUMP IR 图详细信息，输出 inputs_tensor_map、outputs_tensor_map 和 device_matrix。

       不设置或其他值：不打印上述 DUMP IR 相关详细信息。
     -
   * - MS_JIT_DISPLAY_PROGRESS
     - 指定是否打印编译进度的信息。
     - Integer
     - 1：打印关键的编译进度的信息。

       不设置或其他值：不打印编译进度的信息。
     -
   * - MS_DEV_PRECOMPILE_ONLY
     - 指定是否仅预编译网络，而不执行网络。
     - Integer
     - 1：仅预编译网络，而不执行网络。

       不设置或其他值：不预编译网络，即编译并且执行网络。
     -
   * - MS_KERNEL_LAUNCH_SKIP
     - 指定执行过程中需要跳过的算子或者子图
     - String
     - ALL或者all：跳过所有算子和子图的执行

       算子名字（如ReLU）：跳过所有ReLU算子的执行

       子图名字（如kernel_graph_1）：跳过子图kernel_graph_1的执行，用于子图下沉模式
     -
   * - GC_COLLECT_IN_CELL
     - 是否对未使用的Cell对象进行垃圾回收
     - Integer
     - 1：对未使用的Cell对象进行垃圾回收

       不设置或其他值：不会显示调用垃圾回收机制
     - 此环境变量后续将删除，不建议使用。
   * - MS_DEV_USE_PY_BPROP
     - 指定算子的bprop使用python版本，不使用cpp expander
     - String
     - 算子名称，可以指定多个算子，以","分隔
     - 实验性质的环境变量，如果不存在python版本的bprop函数，会执行出错
   * - MS_DEV_DISABLE_BPROP_CACHE
     - 关闭bprop缓存图功能
     - String
     - "on"，表示关闭bprop缓存图功能
     - 实验性质的环境变量，关闭缓存功能会导致构图时间延长
   * - MS_ENABLE_IO_REUSE
     - 开启图输入输出内存复用标志
     - Integer
     - 1: 使能此功能。

       0：不使能。

       默认值：0
     - 仅限Ascend AI处理器环境图编译等级为O2流程使用。
   * - MS_ENABLE_GRACEFUL_EXIT
     - 设置使能进程优雅退出
     - Integer
     - 1：使用进程优雅退出功能。

       不设置或者其他值: 不使用进程优雅退出功能。
     - 使能进程优雅退出功能，依赖callback函数，具体请参考 `进程优雅退出用例 <https://www.mindspore.cn/tutorials/zh-CN/master/train_availability/graceful_exit.html>`_ 。
   * - MS_DEV_BOOST_INFER
     - 针对前端图编译提供编译优化开关。该开关可加速类型推导模块，以加速网络编译。
     - Integer
     - 0: 关闭该优化功能。

       不设置或其他值: 打开该优化功能。
     - 此环境变量后续将删除。

   * - MS_DEV_RUNTIME_CONF
     - 设置运行时控制选项
     - String
     - 配置项，格式为key:value，多个配置项以逗号分隔，例如 `export MS_DEV_RUNTIME_CONF=inline:false,pipeline:false`。

       inline: 子图cell共享场景下，是否开启后端inline，仅在O0或O1模式下生效，默认值为true。

       switch_inline: 是否开启后端控制流inline，仅在O0或O1模式下生效，默认值为true。

       multi_stream: 后端分流方式, 取值可为 1）true 通信计算各一条流。 2）false：关闭多流，通信计算单流。3）group(默认值)：通信算子按照通信域分流。

       pipeline: 是否使能运行时流水，仅在O0或O1模式下生效，默认值为true。

       all_finite: 是否使能溢出检测大算子，仅在O0或O1模式下生效，默认值为true。

       memory_statistics: 是否开启内存统计，默认值为false。

       compile_statistics: 是否开启编译性能统计，默认值为false。

       backend_compile_cache: 是否使用图编译等级O0/O1下的后端编译缓存，仅在前端编译缓存（MS_COMPILER_CACHE_ENABLE）开启时生效，默认值为true。

       view: 是否使能view算子功能，仅在O0或O1模式下生效，默认值为true。
     -
   * - MS_DEV_VIEW_OP
     - 在MS_DEV_RUNTIME_CONF开启view的情况下，指定某些算子进行view替换
     - String
     - 算子名称，可以指定多个算子，以","分隔
     - 实验性质的环境变量

   * - MS_ALLOC_CONF
     - 设置内存策略
     - String
     - 配置项，格式为key:value，多个配置项以逗号分隔，例如 `export MS_ALLOC_CONF=enable_vmm:true,memory_tracker:true`。

       enable_vmm: 是否使能虚拟内存，默认值为true。

       vmm_align_size: 设置虚拟内存对齐大小，单位为MB，默认值为2。

       memory_tracker: 是否开启memory tracker，默认值为false。

       acl_allocator: 是否使用ACL内存分配器，默认值为true。

       somas_whole_block: 是否使用SOMAS整块内存分配，默认值为false。
     -

   * - MS_DEV_GRAPH_KERNEL_FLAGS
     - 设置图算融合的融合策略
     - String
     - 配置项，格式为“--key=value”，多个配置项以空格分隔，多个value以逗号分隔，例如 `export MS_DEV_GRAPH_KERNEL_FLAGS="--enable_expand_ops=Square --enable_cluster_ops=MatMul,Add"`

       opt_level：设置优化级别。默认值： `2` 。

       enable_expand_ops：将不在默认列表的算子强行展开，需有相应算子的expander实现。

       disable_expand_ops：禁止对应算子展开。

       enable_expand_ops_only：仅允许对应算子展开。当设置该选项时，忽略以上两个选项。

       enable_cluster_ops：在默认融合算子名单的基础上，把对应算子加入参与融合的算子集合。

       disable_cluster_ops：禁止对应算子加入参与融合的算子集合。

       enable_cluster_ops_only：仅允许对应算子加入参与融合的算子集合。当设置该选项时，忽略以上两个选项。

       disable_fusion_pattern：禁止对应融合pattern参与融合。

       enable_fusion_pattern_only：仅允许对应融合pattern参与融合。当设置该选项时，忽略以上选项。

       enable_pass：默认关闭的pass可以通过该选项强制使能。

       disable_pass：默认使能的pass可以通过该选项强制关闭。

       dump_as_text：将关键过程的详细信息生成文本文件保存到 `graph_kernel_dump` 目录里。默认值： `False` 。

       enable_debug_mode：在图算kernelmod launch前后插同步，并在launch失败时打印调试信息，仅支持GPU后端。默认值： `False` 。

       path：指定读取json配置。当设置该选项时，忽略以上选项。
     - 详细说明参考 `自定义融合 <https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/fusion_pass.html>`_

Dump调试
--------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - 环境变量
     - 功能
     - 类型
     - 取值
     - 说明
   * - MINDSPORE_DUMP_CONFIG
     - 指定 `云侧Dump功能 <https://www.mindspore.cn/tutorials/zh-CN/master/debug/dump.html>`_
       或 `端侧Dump功能 <https://www.mindspore.cn/lite/docs/zh-CN/master/tools/benchmark_tool.html#dump功能>`_ 所依赖的配置文件的路径
     - String
     - 文件路径，支持相对路径与绝对路径
     -
   * - MS_DIAGNOSTIC_DATA_PATH
     - 使用 `云侧Dump功能 <https://www.mindspore.cn/tutorials/zh-CN/master/debug/dump.html>`_ 时，
       如果Dump配置文件没有设置 `path` 字段或者设置为空字符串，则 `$MS_DIAGNOSTIC_DATA_PATH` `/debug_dump` 就会被当做path的值。
       若Dump配置文件中设置了 `path` 字段，则仍以该字段的实际取值为准。
     - String
     - 文件路径，只支持绝对路径
     - 与MINDSPORE_DUMP_CONFIG配合使用
   * - MINDSPORE_DUMP_IGNORE_USELESS_OUTPUT
     - 是否忽略无用的dump输出，例如Send算子的输出。
     - String
     - "1"：忽略无用的dump输出

       "0"：保留无用的dump输出
     - 默认值："1"。该环境变量仅在MINDSPORE_DUMP_CONFIG配置时生效。
   * - MS_DEV_DUMP_BPROP
     - 在当前路径dump算子反向图的ir文件
     - String
     - "on"，表示在当前路径dump算子反向图的ir文件
     - 实验性质的环境变量
   * - ENABLE_MS_DEBUGGER
     - 是否在训练中启动Debugger
     - Boolean
     - 1：开启Debugger

       0：关闭Debugger
     - 与MS_DEBUGGER_HOST、MS_DEBUGGER_PORT一起使用
   * - MS_DEBUGGER_PARTIAL_MEM
     - 是否开启部分内存复用（只有在Debugger选中的节点才会关闭这些节点的内存复用）
     - Boolean
     - 1：开启Debugger选中节点的内存复用

       0：关闭Debugger选中节点的内存复用
     -
   * - MS_OM_PATH
     - 配置task异常时dump数据路径以及图编译出错时dump的analyze_fail.ir文件的保存目录，保存路径为：指定的路径/rank_${rand_id}/om
     - String
     - 文件路径，支持相对路径与绝对路径
     -
   * - MS_DUMP_SLICE_SIZE
     - 指定Print、TensorDump、TensorSummary、ImageSummary、ScalarSummary、HistogramSummary算子的数据切片大小。
     - Integer
     - 0~2048，单位：MB，默认值为0。当取值为0时，表示不对数据切片。
     -
   * - MS_DUMP_WAIT_TIME
     - 指定Print、TensorDump、TensorSummary、ImageSummary、ScalarSummary、HistogramSummary算子的二阶段超时时间。
     - Integer
     - 0~600，单位：秒，默认值为0。当取值为0时，表示使用默认超时时间，即 `mindspore.get_context("op_timeout")` 的取值。
     - 该环境变量仅仅在MS_DUMP_SLICE_SIZE不为零的情况下生效。目前二阶段的等待时间无法超过mindspore.get_context("op_timeout")的值。

具体用法详见 `Dump功能调试 <https://www.mindspore.cn/tutorials/zh-CN/master/debug/dump.html>`_ 。

分布式并行
-----------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - 环境变量
     - 功能
     - 类型
     - 取值
     - 说明
   * - RANK_ID
     - 指定深度学习时调用Ascend AI处理器的逻辑ID。
     - Integer
     - 0~7，多机并行时不同server中DEVICE_ID会有重复，使用RANK_ID可以避免这个问题（多机并行时 RANK_ID = SERVER_ID * DEVICE_NUM + DEVICE_ID，DEVICE_ID指当前机器的第几个Ascend AI处理器。）
     -
   * - RANK_SIZE
     - 指定深度学习时调用Ascend AI处理器的数量。

       注意：Ascend AI处理器，使用多卡执行分布式用例时，由用户指定。
     - Integer
     - 1~8，调用Ascend AI处理器的数量
     - 与RANK_TABLE_FILE配合使用
   * - RANK_TABLE_FILE 或 MINDSPORE_HCCL_CONFIG_PATH
     - 路径指向文件，包含指定多Ascend AI处理器环境中Ascend AI处理器的 `device_id` 对应的 `device_ip` 。

       注意：Ascend AI处理器，使用多卡执行分布式用例时，由用户指定。
     - String
     - 文件路径，支持相对路径与绝对路径
     - 与RANK_SIZE配合使用
   * - MS_COMM_COMPILER_OPT
     - Ascend后端图模式下编译时，指定可以复用的通信算子的上限。

       注意：Ascend AI处理器，使用多卡执行分布式用例时，由用户指定。
     - Integer
     - -1或正整数：使能通信子图复用，-1表示使用框架默认值，其他正整数表示用户指定值

       不设置或其他值：关闭通信子图复用
     -
   * - DEVICE_ID
     - 昇腾AI处理器的ID，即Device在AI server上的序列号。
     - Integer
     - 昇腾AI处理器的ID，取值范围：[0, 实际Device数量-1]。
     -
   * - MS_ROLE
     - 指定本进程角色。
     - String
     - MS_SCHED: 代表Scheduler进程，一个训练任务只启动一个Scheduler，负责组网，容灾恢复等，不会执行训练代码。

       MS_WORKER: 代表Worker进程，一般设置分布式训练进程为此角色。

       MS_PSERVER: 代表Parameter Server进程，只有在Parameter Server模式下此角色生效。
     - Worker和Parameter Server进程会向Scheduler进程注册从而完成组网。
   * - MS_SCHED_HOST
     - 指定Scheduler的IP地址。
     - String
     - 合法的IP地址。
     - 当前版本暂不支持IPv6地址。
   * - MS_SCHED_PORT
     - 指定Scheduler绑定端口号。
     - Integer
     - 1024～65535范围内的端口号。
     -
   * - MS_NODE_ID
     - 指定本进程的ID，集群内唯一。
     - String
     - 代表本进程的唯一ID，默认由MindSpore自动生成。
     - MS_NODE_ID在在以下情况需要设置，一般情况下无需设置，由MindSpore自动生成：

       开启容灾场景：容灾恢复时需要获取当前进程ID，从而向Scheduler重新注册。

       开启GLOG日志重定向场景：为了保证各训练进程日志独立保存，需设置进程ID，作为日志保存路径后缀。

       指定进程rank id场景：用户可通过设置MS_NODE_ID为某个整数，来指定本进程的rank id。
   * - MS_WORKER_NUM
     - 指定角色为MS_WORKER的进程数量。
     - Integer
     - 大于0的整数。
     - 用户启动的Worker进程数量应当与此环境变量值相等。若小于此数值，组网失败；若大于此数值，Scheduler进程会根据Worker注册先后顺序完成组网，多余的Worker进程会启动失败。
   * - MS_SERVER_NUM
     - 指定角色为MS_PSERVER的进程数量。
     - Integer
     - 大于0的整数。
     - 只在Parameter Server训练模式下需要设置。
   * - MS_INTERFERED_SAPP
     - 开启自动并行SAPP的手自一体功能。
     - Integer
     - 1代表开启，不设置或其他值：关闭。
     -
   * - MS_ENABLE_RECOVERY
     - 开启容灾。
     - Integer
     - 1代表开启，0代表关闭。默认为0。
     -
   * - MS_RECOVERY_PATH
     - 持久化路径文件夹。
     - String
     - 合法的用户目录。
     - Worker和Scheduler进程在执行过程中会进行必要的持久化，如用于恢复组网的节点信息以及训练业务中间状态等，并通过文件保存。
   * - GROUP_INFO_FILE
     - 指定通信域信息存储路径
     - String
     - 通信域信息文件路径，支持相对路径与绝对路径
     -
   * - MS_SIMULATION_LEVEL
     - 指定模拟编译等级。
     - Integer
     - 为0时，不占卡模拟图编译；为1时，不占卡模拟图编译和算子编译；为2时，占卡模拟图编译和算子编译，内存分析更准确；为3时，占卡模拟执行除通信算子以外的算子。默认不开启。
     - 此环境变量主要用于单卡模拟分布式多卡特定rank卡的编译情况，需要RANK_SIZE和RANK_ID配合使用。
   * - DUMP_PARALLEL_INFO
     - 导出自动并行/半自动并行模式下的并行相关通信信息。dump文件路径可以通过环境变量 `MS_DEV_SAVE_GRAPHS_PATH` 设置。
     - Integer
     - 1代表开启该dump功能，其他值或者不设置该环境变量代表关闭。
     - 每张卡保存的json文件包含的字段含义如下：

       hccl_algo: 集合通信算法。

       op_name: 通信算子名称。

       op_type: 通信算子类型。

       shape: 通信算子的shape信息。

       data_type: 通信算子的数据类型。

       global_rank_id: 全局rank编号。

       comm_group_name: 通信算子的通信域名称。

       comm_group_rank_ids: 通信算子的通信域。

       src_rank: Receive算子的对端算子的rank_id。

       dest_rank: Send算子的对端算子的rank_id。

       sr_tag: src和dest相同时，不同send-receive对的标识ID。
   * - MS_CUSTOM_DEPEND_CONFIG_PATH
     - 根据用户指定路径下的配置文件xxx.json插入控制边，在MindSpore中使用原语ops.Depend表达依赖控制关系。
     - String
     - 该环境变量只在Atlas A2系列产品图模式下使能。
     - json文件包含的字段含义如下：

       get_full_op_name_list(bool)：是否生成算子名称列表，可选，默认为false。

       stage_xxx(string)：用于多卡多图场景，即不同的卡执行不同的图（如流水并行），其中stage_xxx只是一个序号标签，序号值没有实际指向意义。

       graph_id(int)：用于区分子图信息，graph_id号需要与实际执行的graph_id一致, 不一致插入控制边的动作将失效。

       depend_src_list(List[string])：需要插入控制边的源端算子名称列表，需要和depend_dest_list中的算子按顺序一一对应，否则插入控制边的动作将失效。

       depend_dest_list(List[string])：需要插入控制边的终端算子名称列表，需要和depend_src_list中的算子按顺序一一对应，否则插入控制边的动作将失效。

       delete_depend_list(List[string])：需要被删除的算子名称列表，算子名称不存在或者和graph_id不匹配，删除节点的动作将失效。


动态组网相关的具体用法详见 `动态组网 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/dynamic_cluster.html>`_ 。

算子编译
--------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - 环境变量
     - 功能
     - 类型
     - 取值
     - 说明
   * - MS_COMPILER_CACHE_ENABLE
     - 表示是否加载或者保存图编译缓存。当 `MS_COMPILER_CACHE_ENABLE` 被设置为 `1` 时，在第一次执行的过程中，一个编译缓存会被生成并且导出为一个MINDIR文件。当该网络被再次执行时，如果 `MS_COMPILER_CACHE_ENABLE` 仍然为 `1` 并且网络脚本没有被更改，那么这个编译缓存会被加载。

       注意：目前只支持有限的Python脚本更改的自动检测，这意味着可能有正确性风险。当前不支持编译后大于2G的图。这是一个实验特性，可能会被更改或者删除。
     - Integer
     - 0：关闭编译缓存功能

       1：开启编译缓存功能
     - 如果与 `MS_COMPILER_CACHE_PATH` 一起使用，编译缓存文件将保存在 `${MS_COMPILER_CACHE_PATH}` `/rank_${RANK_ID}/` 目录下。

       其中 `RANK_ID` 为多卡训练场景中的卡号，单卡场景默认 `RANK_ID=0` 。
   * - MS_COMPILER_CACHE_PATH
     - MindSpore编译缓存目录，存储图和算子编译过程生成的缓存文件，如 `graph_cache` , `kernel_meta` , `somas_meta` 等
     - String
     - 缓存文件路径，支持相对路径与绝对路径
     -
   * - MS_COMPILER_OP_LEVEL
     - Ascend后端编译时，开启debug功能，生成TBE指令映射文件。

       注意：仅Ascend AI处理器环境使用。
     - Integer
     - 0~4，允许设置级别取值范围。

       0：不开启算子debug功能，删除算子编译缓存文件

       1：生成TBE指令映射文件 `*.cce` 和python-cce映射文件 `*_loc.json` ，开启debug功能

       2：生成TBE指令映射文件 `*.cce` 和python-cce映射文件 `*_loc.json` ，开启debug功能，关闭编译优化开关，开启ccec调试功能（ccec编译器选项设置为-O0-g）

       3：不开启算子debug功能，默认值

       4：生成TBE指令映射文件 `*.cce` 和UB融合计算描述文件 `{$kernel_name}_compute.json`
     - 发生AICore Error时，如果需要保存算子cce文件，可以设置 `MS_COMPILER_OP_LEVEL` 为1或2。
   * - MS_ASCEND_CHECK_OVERFLOW_MODE
     - 设置浮点计算结果输出模式
     - String
     - SATURATION_MODE: 饱和模式。

       INFNAN_MODE: INF/NAN模式。

       默认值: INFNAN_MODE。

     - 饱和模式：计算出现溢出时，饱和为浮点数极值（+-MAX）。

       INF/NAN模式：遵循IEEE 754标准，根据定义输出INF/NAN的计算结果。

       仅限Atlas A2训练系列产品使用。
   * - MS_CUSTOM_AOT_WHITE_LIST
     - 指定自定义算子使用动态库的合法路径。
     - String
     - 动态库的合法路径。框架会根据自定义算子使用动态库的合法路径进行校验。当自定义算子使用的动态库不在路径中时，框架会报错并拒绝使用对应动态库。当设置为空时，不对自定义算子动态库进行校验。

       默认：空。
     -

常见问题详见 `FAQ <https://mindspore.cn/docs/zh-CN/master/faq/operators_compile.html>`_ 。

日志
----

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - 环境变量
     - 功能
     - 类型
     - 取值
     - 说明
   * - GLOG_log_dir
     - 指定日志输出的路径
     - String
     - 文件路径，支持相对路径与绝对路径
     - 与 `GLOG_logtostderr` 一起使用

       若 `GLOG_logtostderr` 的值为0，则必须设置此变量

       若指定了 `GLOG_log_dir` 且 `GLOG_logtostderr` 的值为1时，则日志输出到屏幕，不输出到文件

       日志保存路径为： `指定的路径/rank_${rank_id}/logs/` ，非分布式训练场景下， `rank_id` 为0；分布式训练场景下， `rank_id` 为当前设备在集群中的ID

       C++和Python的日志会被输出到不同的文件中，C++日志的文件名遵从 `GLOG` 日志文件的命名规则，这里是 `mindspore.机器名.用户名.log.日志级别.时间戳.进程ID` ，Python日志的文件名为 `mindspore.log.进程ID`

       `GLOG_log_dir` 只能包含大小写字母、数字、"-"、"_"、"/"等字符
   * - GLOG_max_log_size
     - 控制MindSpore C++模块日志单文件大小，可以通过该环境变量更改日志文件默认的最大值
     - Integer
     - 正整数，默认值：50MB
     - 如果当前写入的日志文件超过最大值，则新输出的日志内容会写入到新的日志文件中
   * - GLOG_logtostderr
     - 控制日志的输出方式
     - Integer
     - 1:日志输出到屏幕

       0:日志输出到文件

       默认值：1
     - 与GLOG_log_dir一起使用
   * - GLOG_stderrthreshold
     - 日志模块在将日志输出到文件的同时也会将日志打印到屏幕，GLOG_stderrthreshold用于控制此情况下打印到屏幕的日志级别
     - Integer
     - 0-DEBUG

       1-INFO

       2-WARNING

       3-ERROR

       4-CRITICAL

       默认值：2
     -
   * - GLOG_v
     - 控制日志的级别
     - Integer
     - 0-DEBUG

       1-INFO

       2-WARNING

       3-ERROR，表示程序执行出现报错，输出错误日志，程序可能不会终止

       4-CRITICAL，表示程序执行出现异常，将会终止执行程序

       默认值：2
     - 指定日志级别后，将会输出大于或等于该级别的日志信息
   * - VLOG_v
     - 控制verbose日志的输出，在 `import mindspore` 之前通过export来配置该环境变量
     - String
     - 通过命令：
       `export VLOG_v=20000;python -c 'import mindspore';` 查看MindSpore可用的 verbose 日志级别。

     - 格式1： `VLOG_v=number`，仅输出verbose level值等于 `number` 的日志。

       格式2： `VLOG_v=(number1,number2)`，仅输出verbose level值介于 `number1` 和 `number2` 之间（包含 `number1` 和 `number2`）的日志。特别地， `VLOG_v=(,number2)` 输出 verbose level 介于 `1 ~ number2` 的日志，而 `VLOG_v=(number1,)` 输出 verbose level 介于 `number1 ~ 0x7fffffff` 的日志。

       上面 `number`、 `number1`、 `number2` 的取值只接受非负十进制整数值，最大值取值为 `int` 类型的最大值 `0x7fffffff`。`VLOG_v` 字符串中不能包含空白字符。

       注意：扩号 `()` 对于 `bash` 有特殊含义，当指定范围时，需要用引号包起来，如 `export VLOG_v="(number1,number2)"` 或 `export VLOG_v='(number1,number2)'`。如果直接把环境变量的设置写到命令行中，可以不加引号，如通过命令 `VLOG_v=(1,) python -c 'import mindspore'` 查看 MindSpore 已经使用的 verbose tag 标志。
   * - logger_backupCount
     - 用于控制MindSpore Python模块日志文件数量
     - Integer
     - 默认值：30
     -
   * - logger_maxBytes
     - 用于控制MindSpore Python模块日志单文件大小
     - Integer
     - 默认值：52428800 bytes
     -
   * - MS_SUBMODULE_LOG_v
     - 指定MindSpore C++各子模块的日志级别
     - Dict {String:Integer...}
     - 0-DEBUG

       1-INFO

       2-WARNING

       3-ERROR

     - 赋值方式为：`MS_SUBMODULE_LOG_v="{SubModule1:LogLevel1,SubModule2:LogLevel2,...}"`

       其中被指定子模块的日志级别将覆盖 `GLOG_v` 在此模块内的设置，
       此处子模块的日志级别 `LogLevel` 与 `GLOG_v` 的日志级别含义相同，
       MindSpore子模块列表详见 `sub-module_names <https://gitee.com/mindspore/mindspore/blob/master/mindspore/core/utils/log_adapter.cc>`_。

       例如可以通过 `GLOG_v=1 MS_SUBMODULE_LOG_v="{PARSER:2,ANALYZER:2}"`
       把 `PARSER` 和 `ANALYZER` 模块的日志级别设为WARNING，其他模块的日志级别设为INFO
   * - GLOG_logfile_mode
     - 用于控制MindSpore中GLOG日志文件的权限，是GLOG的环境变量
     - 八进制数字
     - 可参考Linux文件权限设置的数字表示，默认值：0640(取值)
     -
   * - MS_RDR_ENABLE
     - 是否开启程序运行数据记录器（RDR），如果MindSpore出现了运行异常，会自动导出MindSpore中预先记录的数据以辅助定位运行异常的原因
     - Integer
     - 1：开启RDR功能

       0：关闭RDR功能
     - 配合 `MS_RDR_MODE` 与 `MS_RDR_PATH` 使用
   * - MS_RDR_MODE
     - 指定运行数据记录器（RDR）导出数据的模式
     - Integer
     - 1：仅在训练进程异常终止时导出数据

       2：训练进程异常终止或正常结束时导出数据

       默认值：1
     - 配合 `MS_RDR_ENABLE=1` 使用
   * - MS_RDR_PATH
     - 配置程序运行数据记录器（RDR）的文件导出的根目录路径
     - String
     - 目录路径，仅支持绝对路径
     - 配合 `MS_RDR_ENABLE=1` 使用，最终RDR文件将 `${MS_RDR_PATH}` `/rank_${RANK_ID}/rdr/` 目录下。
       其中 `RANK_ID` 为多卡训练场景中的卡号，单卡场景默认 `RANK_ID=0` 。
   * - MS_EXCEPTION_DISPLAY_LEVEL
     - 控制异常信息显示级别
     - Integer
     - 0: 显示与模型开发者和框架开发者相关的异常信息

       1: 显示与模型开发者相关的异常信息

       默认值：0
     -

注意：glog不支持日志文件的绕接，如果需要控制日志文件对磁盘空间的占用，可选用操作系统提供的日志文件管理工具，例如：Linux的logrotate。请在 `import mindspore` 之前设置日志相关环境变量。

特征值检测
------------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - 环境变量
     - 功能
     - 类型
     - 取值
     - 说明
   * - NPU_ASD_ENABLE
     - 是否开启特征值检测功能
     - Integer
     - 0：关闭特征值检测功能

       1：检测到异常，只打印日志，但检测算子不抛异常

       2：检测到异常，打印日志，检测算子抛出异常

       3：特征值正常和异常场景下都会打印（备注：正常场景下只有CANN开启了INFO及DEBUG级别才会打印），检测到异常时检测算子抛出异常
     - 目前本特性仅支持Atlas A2 训练系列产品，仅支持检测Transformer类模型，bfloat16数据类型，训练过程中出现的特征值检测异常

       考虑到无法事先知道数据特征值的分布范围，建议设置NPU_ASD_ENABLE的值为1来使能静默检测，以防止误检导致训练中断
   * - NPU_ASD_UPPER_THRESH
     - 控制检测的绝对数值阈值
     - String
     - 格式为整型数据对，其中第一个元素控制绝对数值一级阈值，第二个元素控制绝对数值二级阈值

       减小阈值可以检出波动更小的异常数据，增加检出率，增大阈值与之相反

       在不配置该环境变量的默认情况下，`NPU_ASD_UPPER_THRESH=1000000,10000`
     -
   * - NPU_ASD_SIGMA_THRESH
     - 控制检测的相对数值阈值
     - String
     - 格式为整型数据对，其中第一个元素控制相对数值一级阈值，第二个元素控制相对数值二级阈值

       减小阈值可以检出波动更小的异常数据，增加检出率，增大阈值与之相反

       在不配置该环境变量的默认情况下，`NPU_ASD_SIGMA_THRESH=100000,5000`
     -

特征值检测的更多内容详见 `特征值检测 <https://www.mindspore.cn/tutorials/zh-CN/master/debug/sdc.html>`_ 。

三方库
------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - 环境变量
     - 功能
     - 类型
     - 取值
     - 说明
   * - OPTION_PROTO_LIB_PATH
     - RPOTO依赖库库路径
     - String
     - 目录路径，支持相对路径与绝对路径
     -
   * - PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION
     - 选择Protocol Buffers后端使用什么语言实现
     - String
     - "cpp"：使用c++后端实现

       "python"：使用python后端实现

       不设置或其他值：使用python后端实现
     -
   * - ASCEND_OPP_PATH
     - OPP包安装路径
     - String
     - OPP包安装的绝对路径
     - 仅限Ascend AI处理器环境需要，一般提供给用户的环境已配置好，无需关心。
   * - ASCEND_AICPU_PATH
     - AICPU包安装路径
     - String
     - AICPU包安装的绝对路径
     - 仅限Ascend AI处理器环境需要，一般提供给用户的环境已配置好，无需关心。
   * - ASCEND_CUSTOM_OPP_PATH
     - 自定义算子包安装路径
     - String
     - 自定义算子包安装的绝对路径
     - 仅限Ascend AI处理器环境需要，一般提供给用户的环境已配置好，无需关心。
   * - ASCEND_TOOLKIT_PATH
     - TOOLKIT包安装路径
     - String
     - 自定义算子包安装的绝对路径
     - 仅限Ascend AI处理器环境需要，一般提供给用户的环境已配置好，无需关心。
   * - CUDA_HOME
     - CUDA安装路径
     - String
     - CUDA包安装的绝对路径
     - 仅限GPU环境需要，一般无需设置，如在GPU环境中安装了多种版本的CUDA，为了避免混淆，建议配置此环境变量。
   * - MS_ENABLE_TFT
     - 使能 `MindIO TFT <https://www.hiascend.com/document/detail/zh/mindx-dl/600/clusterscheduling/ref/mindiottp/mindiotft001.html>`_ 特性，表示启用 TTP、UCE、TRE 或 ARF 功能。
     - String
     - "{TTP:1,UCE:1,ARF:1}"。TTP (Try To Persist)：临终 CKPT 功能、UCE (Uncorrectable Memory Error)：UCE 故障容错恢复功能、TRE (Training Result Error)：训练结果异常恢复功能、ARF (Air Refuelling)：进程级重调度恢复功能。四个特性可以分开使能，如果只想启用其中的某一个功能，则将对应的值设置为 1 即可。其他值：未开启MindIO TFT。（开启 UCE 或者 ARF 功能时，默认开启 TTP 功能。TRE 功能不可以与 UCE 或 ARF 功能同时使用。）
     - 仅限在 Ascend 后端开启图模式，且 jit_level 设置为 "O0" 或 "O1"。
   * - MS_TFT_IP
     - MindIO的controller线程所在IP，供processor链接。
     - String
     - IP地址。
     - 仅限在 Ascend 后端开启图模式，且 jit_level 设置为 "O0" 或 "O1"。
   * - MS_TFT_PORT
     - MindIO的controller线程绑定端口，供processor链接。
     - Integer
     - 正整数。
     - 仅限在 Ascend 后端开启图模式，且 jit_level 设置为 "O0" 或 "O1"。
   * - AITURBO
     - 使能华为云存储加速
     - String
     - "1": 使能华为云存储加速。 其他值：关闭华为云存储加速。 默认值：空。
     - 仅限华为云环境。

CANN
--------

CANN的环境变量详见 `昇腾社区 <https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/developmentguide/appdevg/aclpythondevg/aclpythondevg_02_0004.html>`_ 。请在 `import mindspore` 之前设置CANN的环境变量。

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - 环境变量
     - 功能
     - 类型
     - 取值
     - 说明
   * - MS_FORMAT_MODE
     - 设置Ascend 图编译等级为O2流程的默认优选格式，整网设置为ND格式
     - Integer
     - 1: 算子优先选择ND格式。

       0：算子优先选择私有格式。

       默认值：1。
     - 此环境变量影响算子的format选择，从而对网络执行性能和内存占用产生影响，可通过设置此选项测试得到性能和内存更优的算子格式选择。

       仅限Ascend AI处理器环境图编译等级为O2流程使用。

Profiler
-----------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - 环境变量
     - 功能
     - 类型
     - 取值
     - 说明
   * - MS_PROFILER_OPTIONS
     - 设置Profiler的配置信息
     - String
     - 配置Profiler的采集选项，格式为JSON字符串。其中以下几个参数类型与实例化Profiler方式有差异，取值含义相同：

       activities (list, 可选) - 设置采集性能数据的设备，可传多个设备，默认值：[CPU, NPU]。可取值：[CPU]、[NPU]、[CPU, NPU]。

       aic_metrics (str, 可选) - 设置AI Core指标类型。默认值：AicoreNone。可取值：AicoreNone、ArithmeticUtilization、PipeUtilization、Memory、MemoryL0、ResourceConflictRatio、MemoryUB、L2Cache、MemoryAccess。

       profiler_level (str, 可选) - 设置采集性能数据级别。默认值：Level0。可取值：Level0、Level1、Level2。

       其他参数可参考 `MindSpore profile参数详解 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.profiler.profile.html>`_ 。

     - 此环境变量使能与输入参数实例化Profiler方式使能性能数据采集的方式二选一。
   * - PROFILING_MODE
     - 设置CANN Profiling的模式
     - String
     - true：开启Profiling功能。

       false或者不配置：关闭Profiling功能。

       dynamic：动态采集性能数据模式。

     - 此环境变量为CANN Profiling使能环境变量，Profiler读取此环境变量用于检查避免重复开启CANN Profiling。用户不需要手动设置此环境变量。

动态图
-----------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - 环境变量
     - 功能
     - 类型
     - 取值
     - 说明
   * - MS_PYNATIVE_CONFIG_STATIC_SHAPE
     - 动态图模式反向整图下发开关。
     - String
     - '1'：开启反向整图执行开关。
       不设置或其他值：关闭该功能。
     - 开启该功能后，动态图反向会通过整图下发。

源码构建
-----------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - 环境变量
     - 功能
     - 类型
     - 取值
     - 说明
   * - MSLIBS_CACHE_PATH
     - MindSpore源码编译时，编译的第三方库的安装路径。
     - String
     - "~/.mslib": 源码编译过程中编译的第三方库的安装位置。默认值：空。
     - 设置该变量后，MindSpore源码编译过程中编译的第三方库会被安装到变量指定的目录下，从而支持在多次编译间共享第三方库，大幅降低编译耗时。
   * - MSLIBS_SERVER
     - MindSpore源码编译时，从该变量指向的地址下载第三方库源码。
     - String
     - "tools.mindspore.cn"：MindSpore官方下载源。默认值：空。
     - 设置该变量后，MindSpore源码编译时会从变量指向的路径下载第三方库源代码，避免访问github的网络不稳定问题，提升下载速度。该变量在编译选项包含-S on时不生效。
