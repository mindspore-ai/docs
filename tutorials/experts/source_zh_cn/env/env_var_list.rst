环境变量
========

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png 
   :target: https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/env/env_var_list.rst

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
       
       注意：仅Ascend AI处理器环境使用。
     - Integer
     - 1~24：允许设置并行进程数取值范围
     - 
   * - MS_COMPILER_CACHE_ENABLE
     - 指定是否保存和加载前端的图编译缓存。该功能与 mindspore context 中的 `enable_compile_cache <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_context.html#mindspore.set_context>`_ 相同。

       注意：该环境变量优先级低于 `enable_compile_cache` context。
     - Integer
     - 0：关闭前端图编译缓存功能

       1：开启前端图编译缓存功能
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

       3：不开启算子debug功能

       4：生成TBE指令映射文件 `*.cce` 和UB融合计算描述文件 `{$kernel_name}_compute.json`
     - 
   * - MS_DEV_DISABLE_PREBUILD
     - Ascend后端编译时，关闭算子预编译，默认不设置此环境变量。算子预编译可能会修正算子注册的fusion_type属性进而影响到算子融合，如遇到融合算子性能较差时，可尝试开启此环境变量验证是否是融合算子本身问题。

       注意：仅Ascend AI处理器环境使用。
     - Boolean
     - true：关闭预编译 

       false：使能预编译
     - 

具体用法详见 `算子增量编译 <https://mindspore.cn/tutorials/experts/zh-CN/master/debug/op_compilation.html>`_ ，常见问题详见 `FAQ <https://mindspore.cn/docs/zh-CN/master/faq/operators_compile.html>`_ 。

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
     - 0~7，多机并行时不同server中DEVICE_ID会有重复，使用RANK_ID可以避免这个问题（多机并行时 RANK_ID = SERVER_ID * DEVICE_NUM + DEVICE_ID）
     - 
   * - RANK_SIZE
     - 指定深度学习时调用Ascend AI处理器的数量。

       注意：Ascend AI处理器，使用多卡执行分布式用例时，由用户指定。
     - Integer
     - 1~8，调用Ascend AI处理器的数量
     - 与RANK_TABLE_FILE配合使用
   * - RANK_TABLE_FILE
     - 路径指向文件，包含指定多Ascend AI处理器环境中Ascend AI处理器的 `device_id` 对应的 `device_ip` 。

       注意：Ascend AI处理器，使用多卡执行分布式用例时，由用户指定。
     - String
     - 文件路径，支持相对路径与绝对路径
     - 与RANK_SIZE配合使用

具体用法详见 `分布式并行训练基础样例 <https://mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_ascend.html#运行脚本>`_ 。

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
     - 配合 `MS_RDR_ENABLE=1` 使用，最终RDR文件将 `${MS_RDR_PATH}` `/rank_${RANK_ID}/rdr/`目录下。
       其中 `RANK_ID` 为多卡训练场景中的卡号，单卡场景默认 `RANK_ID=0` 。

具体用法详见 `Running Data Recorder <https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/custom_debug.html#running-data-recorder>`_ 。

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
     - 与GLOG_logtostderr一起使用
   * - GLOG_log_max
     - 控制MindSpore C++模块日志单文件大小
     - Integer
     - 正整数，默认值：50
     - 
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

       默认值：2
     - 
   * - GLOG_v
     - 控制日志的级别
     - Integer
     - 0-DEBUG
       
       1-INFO

       2-WARNING

       3-ERROR

       默认值：2
     - 
   * - logger_backupCount
     - 用于控制MindSpore Python模块日志文件数量
     - Integer
     - 默认值：30
     - 
   * - logger_maxBytes
     - 用于控制MindSpore Python模块日志单文件大小
     - Integer
     - 默认值：52428800
     - 
   * - MS_SUBMODULE_LOG_v
     - 指定MindSpore C++各子模块的日志级别
     - Dict {String:Integer...}
     - 0-DEBUG
       
       1-INFO

       2-WARNING

       3-ERROR
       
       SubModule: COMMON, MD, DEBUG, DEVICE, COMMON, IR...
     - 

具体用法详见 `日志功能与用法 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/custom_debug.html#日志相关的环境变量和配置>`_ 。

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
     - 是否开启numa绑核功能，在大多数分布式场景下numa绑核都能提升数据处理效率和端到端性能
     - String
     - True: 开启numa绑核功能
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
     - MindInsight Debugger服务的IP
     - String
     - 启动MindInsight调试器的机器的IP
     - 与ENABLE_MS_DEBUGGER=1、MS_DEBUGGER_PORT一起使用
   * - MS_DEBUGGER_PARTIAL_MEM
     - 是否开启部分内存复用（只有在Debugger选中的节点才会关闭这些节点的内存复用）
     - Boolean
     - 1：开启Debugger选中节点的内存复用

       0：关闭Debugger选中节点的内存复用
     - 
   * - MS_DEBUGGER_PORT
     - 连接MindInsight Debugger Server的端口
     - Integer
     - 1~65536，连接MindInsight Debugger Server的端口
     - 与ENABLE_MS_DEBUGGER=1、MS_DEBUGGER_HOST一起使用

具体用法详见 `调试器 <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/debugger.html>`_ 。

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
   * - MS_DEV_ENABLE_FALLBACK
     - 设置非0值时使能Fallback功能
     - Integer
     - 1: 开启Fallback功能

       0: 关闭Fallback功能

       默认值：1
     - 
   * - MS_EXCEPTION_DISPLAY_LEVEL
     - 控制异常信息显示级别
     - Integer
     - 0: 显示与模型开发者和框架开发者相关的异常信息

       1: 显示与模型开发者相关的异常信息

       默认值：0
     - 
   * - MS_OM_PATH
     - 配置task异常时dump数据路径以及图编译出错时dump的analyze_fail.dat文件的保存目录，保存路径为：指定的路径/rank_${rand_id}/om
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
   * - MS_DEV_SAVE_GRAPTHS_SORT_MODE
     - 选择生成ir文件的图打印排序方式
     - Integer
     - 0: 打印默认ir文件

       1: 打印异序ir文件
     - 