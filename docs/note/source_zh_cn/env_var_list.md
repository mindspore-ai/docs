# 环境变量

`Linux` `Ascend` `GPU` `CPU` `初级` `中级` `高级`

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/note/source_zh_cn/env_var_list.md)

本文介绍MindSpore的环境变量。

| 环境变量 | 所属模块 | 功能 | 类型 | 取值范围 | 配置关系 | 是否必选 |
| --- | --- | --- | --- | --- | --- | --- |
|MS_ENABLE_CACHE|MindData|是否开启dataset数据处理cache功能，可以实现数据处理过程中数据的cache能力，加速数据集读取及增强处理|String|TRUE：开启数据处理cache功能 <br>FALSE：关闭数据处理cache功能|与MS_CACHE_HOST、MS_CACHE_PORT一起使用|可选|
|MS_CACHE_HOST|MindData|开启cache时，cache服务所在的IP|String|Cache Server所在机器的IP|与MS_ENABLE_CACHE=TRUE、MS_CACHE_PORT一起使用|可选|
|MS_CACHE_PORT|MindData|开启cache时，cache服务所在的端口|String|Cache Server所在机器的端口|与MS_ENABLE_CACHE=TRUE、MS_CACHE_HOST一起使用|可选|
|PROFILING_MODE|MindData|是否开启dataset profiling数据处理性能分析，用于与MindInsight一起配合使用，可以在网页中展示各个阶段的耗时|String|true: 开启profiling功能<br>false: 关闭profiling功能|与MINDDATA_PROFILING_DIR配合使用|可选|
|MINDDATA_PROFILING_DIR|MindData|系统路径，保存dataset profiling结果路径|String|系统路径，支持相对路径|与PROFILING_MODE=true配合使用|可选|
|DATASET_ENABLE_NUMA|MindData|是否开启numa绑核功能，在大多数分布式场景下numa绑核都能提升数据处理效率和端到端性能|String|True: 开启numa绑核功能|与libnuma.so配合使用|可选|
|OPTIMIZE|MindData|是否执行dataset数据处理 pipeline 树优化，在适合数据处理算子融合的场景下，可以提升数据处理效率|String|true: 开启pipeline树优化<br>false: 关闭pipeline树优化|无|可选|
|ENABLE_MS_DEBUGGER|Debugger|是否在训练中启动Debugger|Boolean|1：开启Debugger<br>0：关闭Debugger|无|可选|
|MS_DEBUGGER_PORT|Debugger|连接MindInsight Debugger Server的端口|Integer|1~65536，连接MindInsight Debugger Server的端口|无|可选
|MS_DEBUGGER_PARTIAL_MEM|Debugger|是否开启部分内存复用（只有在Debugger选中的节点才会关闭这些节点的内存复用）|Boolean|1：开启Debugger选中节点的内存复用<br>0：关闭Debugger选中节点的内存复用|无|可选|
|MS_BUILD_PROCESS_NUM|MindSpore|Ascend后端编译时，指定并行编译进程数|Integer|1~24：允许设置并行进程数取值范围|无|可选|
|RANK_TABLE_FILE|MindSpore|路径指向文件，包含指定多Ascend AI处理器环境中Ascend AI处理器的"device_id"对应的"device_ip"。|String|文件路径，支持相对路径与绝对路径|与RANK_SIZE配合使用|必选（使用Ascend AI处理器时）|
|RANK_SIZE|MindSpore|指定深度学习时调用Ascend AI处理器的数量|Integer|1~8，调用Ascend AI处理器的数量|与RANK_TABLE_FILE配合使用|必选（使用Ascend AI处理器时）|
|RANK_ID|MindSpore|指定深度学习时调用Ascend AI处理器的逻辑ID|Integer|0~7，多机并行时不同server中DEVICE_ID会有重复，使用RANK_ID可以避免这个问题（多机并行时 RANK_ID = SERVER_ID * DEVICE_NUM + DEVICE_ID|无|可选|
|MS_SUBMODULE_LOG_v|MindSpore|[MS_SUBMODULE_LOG_v功能与用法](https://www.mindspore.cn/tutorial/training/zh-CN/r1.2/advanced_use/custom_debugging_info.html#id9)|Dict{String:Integer...}|LogLevel: 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR<br>SubModual: COMMON, MD, DEBUG, DEVICE, COMMON, IR...|无|可选
|OPTION_PROTO_LIB_PATH|MindSpore|RPOTO依赖库库路径|String|目录路径，支持相对路径与绝对路径|无|可选|
|MS_RDR_ENABLE|MindSpore|是否开启程序运行数据记录器（RDR），如果MindSpore出现了运行异常，会自动导出MindSpore中预先记录的数据以辅助定位运行异常的原因|Integer|1：开启RDR功能 <br>0：关闭RDR功能|与MS_RDR_PATH一起使用|可选|
|MS_RDR_PATH|MindSpore|配置程序运行数据记录器（RDR）的文件导出路径|String|文件路径，仅支持绝对路径|与MS_RDR_ENABLE=1一起使用|可选|
|GE_USE_STATIC_MEMORY|GraphEngine|当网络模型层数过大时，特征图中间计算数据可能超过25G，例如BERT24网络。多卡场景下为保证通信内存高效协同，需要配置为1，表示使用内存静态分配方式，其他网络暂时无需配置，默认使用内存动态分配方式。<br>静态内存默认配置为31G，如需要调整可以通过网络运行参数graph_memory_max_size和variable_memory_max_size的总和指定；动态内存是动态申请，最大不会超过graph_memory_max_size和variable_memory_max_size的总和。|Integer|1：使用内存静态分配方式<br>0：使用内存动态分配方式|无|可选|
|DUMP_GE_GRAPH|GraphEngine|把整个流程中各个阶段的图描述信息打印到文件中，此环境变量控制dump图的内容多少|Integer|1：全量dump<br>2：不含有权重等数据的基本版dump<br>3：只显示节点关系的精简版dump|无|可选|
|DUMP_GRAPH_LEVEL|GraphEngine|把整个流程中各个阶段的图描述信息打印到文件中，此环境变量可以控制dump图的个数|Integer|1：dump所有图<br>2：dump除子图外的所有图<br>3：dump最后的生成图|无|可选|
