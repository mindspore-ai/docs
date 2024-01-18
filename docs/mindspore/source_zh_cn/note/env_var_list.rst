环境变量
========

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg
    :target: https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/env_var_list.rst
    :alt: 查看源文件

本文介绍MindSpore的环境变量。

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
   * - MS_BUILD_PROCESS_NUM
     - Ascend后端编译时，指定并行编译进程数。

     - Integer
     - 1~24：允许设置并行进程数取值范围
     -
   * - MS_COMPILER_CACHE_ENABLE
     - 指定是否保存和加载编译缓存。该功能与 mindspore context 中的 `enable_compile_cache <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_context.html#mindspore.set_context>`_ 相同。

       注意：该环境变量优先级低于 `enable_compile_cache` context。
     - Integer
     - 0：关闭编译缓存功能

       1：开启编译缓存功能
     - 如果与 `MS_COMPILER_CACHE_PATH` 一起使用，编译缓存文件将保存在 `${MS_COMPILER_CACHE_PATH}` `/rank_${RANK_ID}/graph_cache/` 目录下。

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
   * - MS_DEV_DISABLE_PREBUILD
     - Ascend后端编译时，关闭算子预编译，默认不设置此环境变量。算子预编译可能会修正算子注册的fusion_type属性进而影响到算子融合，如遇到融合算子性能较差时，可尝试开启此环境变量验证是否是融合算子本身问题。

     - Boolean
     - true：关闭预编译

       false：使能预编译
     -
   * - MINDSPORE_OP_INFO_PATH
     - 指定算子信息库加载文件路径
     - string
     - 文件绝对路径

       默认：不设置。
     - 仅推理使用

具体用法详见 `算子增量编译 <https://mindspore.cn/tutorials/experts/zh-CN/master/optimize/op_compilation.html>`_ ，常见问题详见 `FAQ <https://mindspore.cn/docs/zh-CN/master/faq/operators_compile.html>`_ 。

并行训练
--------

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

动态组网
--------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - 环境变量
     - 功能
     - 类型
     - 取值
     - 说明
   * - MS_ROLE
     - 指定本进程角色。
     - String
     - MS_SCHED: 代表Scheduler进程，一个训练任务只启动一个Scheduler，负责组网，容灾恢复等，不会执行训练代码。

       MS_WORKER: 代表Worker进程，一般设置分布式训练进程为此角色。

       MS_PSERVER: 代表Parameter Server进程，只有在Parameter Server模式下此角色生效，具体请参考 `Parameter Server模式 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/parameter_server_training.html>`_ 。
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
   * - MS_HCCL_CM_INIT
     - 是否使用CM方式初始化HCCL。
     - Integer
     - 1代表是，0代表否。默认为0。
     - 此环境变量只在Ascend硬件平台并且通信域数量较多的情况下建议开启。开启此环境变量后，能够降低HCCL集合通信库的内存占用，并且训练任务执行方式与rank table启动方式相同。

具体用法详见 `动态组网 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/dynamic_cluster.html>`_ 。

运行数据保存
------------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - 环境变量
     - 功能
     - 类型
     - 取值
     - 说明
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

具体用法详见 `Running Data Recorder <https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/rdr.html#running-data-recorder>`_ 。

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

注意：glog不支持日志文件的绕接，如果需要控制日志文件对磁盘空间的占用，可选用操作系统提供的日志文件管理工具，例如：Linux的logrotate。请在 `import mindspore` 之前设置日志相关环境变量。

Dump功能
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
     - 指定 `云侧Dump功能 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/dump.html#同步dump>`_
       或 `端侧Dump功能 <https://www.mindspore.cn/lite/docs/zh-CN/master/use/benchmark_tool.html#dump功能>`_ 所依赖的配置文件的路径
     - String
     - 文件路径，支持相对路径与绝对路径
     -
   * - MS_DIAGNOSTIC_DATA_PATH
     - 使用 `云侧Dump功能 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/dump.html#同步dump>`_ 时，
       如果Dump配置文件没有设置 `path` 字段或者设置为空字符串，则 `$MS_DIAGNOSTIC_DATA_PATH` `/debug_dump` 就会被当做path的值。
       若Dump配置文件中设置了 `path` 字段，则仍以该字段的实际取值为准。
     - String
     - 文件路径，只支持绝对路径
     - 与MINDSPORE_DUMP_CONFIG配合使用
   * - MS_DEV_DUMP_BPROP
     - 在当前路径dump算子反向图的ir文件
     - String
     - "on"，表示在当前路径dump算子反向图的ir文件
     - 实验性质的环境变量
   * - MS_DEV_DUMP_PACK
     - 在当前路径生成trace构图的ir文件
     - String
     - "on"，表示在当前路径生成trace构图的ir文件
     - 实验性质的环境变量

具体用法详见 `Dump功能调试 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/dump.html>`_ 。

数据处理性能
------------

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
   * - MS_ENABLE_NUMA
     - 是否开启全局numa绑核功能，提升端到端性能
     - String
     - True: 开启全局numa绑核功能
     -
   * - OPTIMIZE
     - 是否执行dataset数据处理 pipeline 树优化，在适合数据处理算子融合的场景下，可以提升数据处理效率
     - String
     - true: 开启pipeline树优化

       false: 关闭pipeline树优化
     -

具体用法详见 `单节点数据缓存 <https://mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_
和 `数据处理性能优化 <https://mindspore.cn/tutorials/experts/zh-CN/master/dataset/optimize.html>`_ 。

调试器
------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - 环境变量
     - 功能
     - 类型
     - 取值
     - 说明
   * - ENABLE_MS_DEBUGGER
     - 是否在训练中启动Debugger
     - Boolean
     - 1：开启Debugger

       0：关闭Debugger
     - 与MS_DEBUGGER_HOST、MS_DEBUGGER_PORT一起使用
   * - MS_DEBUGGER_HOST
     - MindSpore Insight Debugger服务的IP
     - String
     - 启动MindSpore Insight调试器的机器的IP
     - 与ENABLE_MS_DEBUGGER=1、MS_DEBUGGER_PORT一起使用
   * - MS_DEBUGGER_PARTIAL_MEM
     - 是否开启部分内存复用（只有在Debugger选中的节点才会关闭这些节点的内存复用）
     - Boolean
     - 1：开启Debugger选中节点的内存复用

       0：关闭Debugger选中节点的内存复用
     -
   * - MS_DEBUGGER_PORT
     - 连接MindSpore Insight Debugger Server的端口
     - Integer
     - 1~65536，连接MindSpore Insight Debugger Server的端口
     - 与ENABLE_MS_DEBUGGER=1、MS_DEBUGGER_HOST一起使用

具体用法详见 `调试器 <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/debugger.html>`_ 。

网络编译
--------

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
     - 模块名，对应import导入的顶层模块的名称。如果有多个，使用英文逗号分隔。例如：`export MS_JIT_MODULES=mindflow,mindyolo`。
     - 默认情况下，第三方库之外的模块都会进行JIT静态编译。MindSpore套件等一些模块如 `mindflow`、`mindyolo` 等并不会被视作第三方库，请参考 `调用第三方库 <https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html#%E8%B0%83%E7%94%A8%E7%AC%AC%E4%B8%89%E6%96%B9%E5%BA%93>`_ 。如果有类似MindSpore套件的模块，内部存在 `nn.Cell`、`@ms.jit` 修饰函数或需要编译成静态计算图的函数方法，可以通过配置该环境变量，使该模块进行JIT静态编译而不会被当成第三方库。
   * - MS_JIT_IGNORE_MODULES
     - 指定静态图模式下哪些模块是第三方库，不进行JIT静态编译，其函数方法会被解释执行。
     - String
     - 模块名，对应import导入的顶层模块的名称。如果有多个，使用英文逗号分隔。例如：`export MS_JIT_IGNORE_MODULES=numpy,scipy`。
     - 静态图模式能够自动识别第三方库，一般情况下不需要为NumPy、SciPy这些可识别的第三方库设置该环境变量。如果 `MS_JIT_IGNORE_MODULES` 和 `MS_JIT_MODULES` 同时指定同一个模块名，前者生效，后者不生效。
   * - MS_DEV_FALLBACK_DUMP_NODE
     - 是否打印代码中由 `静态图语法增强技术 <https://www.mindspore.cn/docs/zh-CN/master/design/dynamic_graph_and_static_graph.html#%E9%9D%99%E6%80%81%E5%9B%BE%E8%AF%AD%E6%B3%95%E5%A2%9E%E5%BC%BA%E6%8A%80%E6%9C%AF>`_ 支持的语法表达式
     - Integer
     - 1：开启打印功能。

       不设置或其它值：关闭打印功能。
     -
   * - MS_JIT
     - 是否使用JIT即时编译
     - Integer
     - 0：不使用JIT即时编译，网络脚本直接按照动态图（PyNative）模式执行。

       不设置或其它值：根据网络脚本判断执行静态图（Graph）模式还是动态图（PyNative）模式。
     -
   * - MS_DEV_FORCE_USE_COMPILE_CACHE
     - 是否直接使用编译缓存，不检查网络脚本有无被修改
     - Integer
     - 1：不检查网络脚本是否被修改，直接读取编译缓存。建议只在调试过程中使用，例如网络脚本只增加了print语句用于打印调试。

       不设置或其它值：检测网络脚本的改动，网络没有被修改时，才读取编译缓存。
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
     - 0：除return节点外，只打印节点的operator和operand，并且简化子图的打印信息。

       1：打印除debug info和scope以外的所有信息。

       2或不设置：打印所有信息。
     -
   * - MS_DEV_DUMP_IR_INTERVAL
     - 设置间隔多少个IR文件打印保存一个IR文件，减少IR图的打印数量。
     - Integer
     - 1或不设置：打印保存所有IR文件。

       其它数值：按照指定的间隔个数保存IR文件。
     -
   * - MS_DEV_DUMP_IR_PASSES
     - 根据文件名指定保存哪些IR文件。
     - String
     - 文件名或文件名的一部分。如果有多个，使用逗号隔开。例如`export MS_DEV_DUMP_IR_PASSES=recompute,renormalize`。
     -

CANN
--------

CANN的环境变量详见 `昇腾社区 <https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/reference/envvar/envref_07_0001.html>`_ 。请在 `import mindspore` 之前设置CANN的环境变量。


其他
----

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - 环境变量
     - 功能
     - 类型
     - 取值
     - 说明
   * - GROUP_INFO_FILE
     - 指定通信域信息存储路径
     - String
     - 通信域信息文件路径，支持相对路径与绝对路径
     -
   * - GRAPH_OP_RUN
     - 图模式下以任务下沉方式运行pipeline大网络模型时，可能会由于流资源限制而无法正常启动，此环境变量可以指定图模式的执行方式，配置为0表示任务下沉，是默认执行方式；1则表示非任务下沉方式，该方式没有流的限制，但性能有所下降。
     - Integer
     - 0：执行任务下沉

       1：执行非任务下沉
     -
   * - MS_EXCEPTION_DISPLAY_LEVEL
     - 控制异常信息显示级别
     - Integer
     - 0: 显示与模型开发者和框架开发者相关的异常信息

       1: 显示与模型开发者相关的异常信息

       默认值：0
     -
   * - MS_OM_PATH
     - 配置task异常时dump数据路径以及图编译出错时dump的analyze_fail.ir文件的保存目录，保存路径为：指定的路径/rank_${rand_id}/om
     - String
     - 文件路径，支持相对路径与绝对路径
     -
   * - OPTION_PROTO_LIB_PATH
     - RPOTO依赖库库路径
     - String
     - 目录路径，支持相对路径与绝对路径
     -
   * - MS_KERNEL_LAUNCH_SKIP
     - 指定执行过程中需要跳过的算子或者子图
     - String
     - ALL或者all：跳过所有算子和子图的执行

       算子名字（如ReLU）：跳过所有ReLU算子的执行

       子图名字（如kernel_graph_1）：跳过子图kernel_graph_1的执行，用于子图下沉模式
     -
   * - MS_PYNATIVE_GE
     - 设置动态图模式下是否执行GE
     - Integer
     - 0: 不执行GE。

       1: 执行GE。

       默认值: 0
     - 实验性质的环境变量
   * - GC_COLLECT_IN_CELL
     - 是否对未使用的Cell对象进行垃圾回收
     - Integer
     - 1：对未使用的Cell对象进行垃圾回收

       不设置或其他值：不会显示调用垃圾回收机制
     -
   * - PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION
     - 选择Protocol Buffers后端使用什么语言实现
     - String
     - "cpp"：使用c++后端实现

       "python"：使用python后端实现

       不设置或其他值：使用python后端实现
     -
   * - MS_DEV_DISABLE_BPROP_CACHE
     - 关闭bprop缓存图功能
     - String
     - "on"，表示关闭bprop缓存图功能
     - 实验性质的环境变量，关闭缓存功能会导致构图时间延长
   * - MS_DEV_USE_PY_BPROP
     - 指定算子的bprop使用python版本，不使用cpp expander
     - String
     - 算子名称，可以指定多个算子，以","分隔
     - 实验性质的环境变量，如果不存在python版本的bprop函数，会执行出错
   * - MS_DEV_DISABLE_TRACE
     - 关闭trace构图功能
     - String
     - "on"，表示关闭trace构图功能
     - 实验性质的环境变量
   * - MS_FORMAT_MODE
     - 设置Ascend GE流程的默认优选格式，整网设置为ND格式
     - Integer
     - 1: 算子优先选择ND格式。

       0：算子优先选择私有格式。

       默认值：1。
     - 此环境变量影响算子的format选择，从而对网络执行性能和内存占用产生影响，可通过设置此选项测试得到性能和内存更优的算子格式选择。

       仅限Ascend AI处理器环境GE流程使用。
   * - MS_ENABLE_IO_REUSE
     - 开启图输入输出内存复用标志
     - Integer
     - 1: 使能此功能。

       0：不使能。

       默认值：0
     - 仅限Ascend AI处理器环境GE流程使用。
   * - MS_ASCEND_CHECK_OVERFLOW_MODE
     - 设置浮点计算结果输出模式
     - String
     - SATURATION_MODE: 饱和模式。

       INFNAN_MODE: INF/NAN模式。

       默认值: INFNAN_MODE。

     - 饱和模式：计算出现溢出时，饱和为浮点数极值（+-MAX）。

       INF/NAN模式：遵循IEEE 754标准，根据定义输出INF/NAN的计算结果。

       仅限Atlas A2训练系列产品使用。
   * - MS_DISABLE_REF_MODE
     - 设置强制关闭ref模式
     - Integer
     - 0: 不关闭ref模式。

       1: 强制关闭ref模式。

       默认值: 0。

     - 此环境变量后续将删除，不建议使用。

       仅限Ascend AI处理器环境GE流程使用。
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
