# 环境变量

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/note/source_zh_cn/env_var_list.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本文介绍MindSpore的环境变量。

| 环境变量 | 所属模块 | 功能 | 类型 | 取值范围 | 配置关系 | 是否必选 | 默认值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
|MS_BUILD_PROCESS_NUM|MindSpore|Ascend后端编译时，指定并行编译进程数|Integer|1~24：允许设置并行进程数取值范围|无|可选（仅Ascend AI处理器环境使用）|无|
|MS_COMPILER_CACHE_PATH|MindSpore|MindSpore编译缓存目录，存储图和算子编译过程生成的缓存文件，如`graph_cache`,`kernel_meta`，`somas_meta`等|String|缓存文件路径，支持相对路径与绝对路径|无|可选|无|
|MS_COMPILER_CACHE_ENABLE|MindSpore|指定是否保存和加载前端的图编译缓存。该功能与mindspore context中的[enable_compile_cache](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.context.html#mindspore.context.set_context)相同。<br>注意：该环境变量优先级低于`enable_compile_cache` context。|Integer|0: 关闭前端图编译缓存功能<br>1: 开启前端图编译缓存功能|如果与`MS_COMPILER_CACHE_PATH`一起使用，编译缓存文件将保存在`${MS_COMPILER_CACHE_PATH}/rank_${RANK_ID}/graph_cache/`目录下。其中`RANK_ID`为多卡训练场景中的卡号，单卡场景默认`RANK_ID=0`。|可选|无|
|MS_COMPILER_OP_LEVEL|MindSpore|Ascend后端编译时，开启debug功能，生成TBE指令映射文件|Integer|0或1，允许设置级别取值范围。0：不开启算子debug功能。1：生成TBE指令映射文件（cce文件*.cce和python-cce映射文件*_loc.json，同时关闭编译优化开关）|无|可选（仅Ascend AI处理器环境使用）|无|
|MS_DISABLE_PREBUILD|MindSpore|Ascend后端编译时，关闭算子预编译，默认不设置此环境变量。算子预编译可能会修正算子注册的fusion_type属性进而影响到算子融合，如遇到融合算子性能较差时，可尝试开启此环境变量验证是否是融合算子本身问题|Boolean|true：关闭预编译，false：使能预编译|无|可选（仅Ascend AI处理器环境使用）|无|
|MS_GRAPH_KERNEL_FLAGS|MindSpore|图算融合功能的控制选项，可用来开启或关闭图算融合功能、支持对图算融合功能中若干优化的精细控制、支持dump图算融合时的过程数据，用于问题定位和性能调优<br>注意：此环境变量从1.6版本起弃用，后续版本将会删除，请优先使用context中的`graph_kernel_flags`|String|格式和功能同mindspore/context.py中[graph_kernel_flags](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.context.html#mindspore.context.set_context)。<br>注：环境变量优先级高于context，即，若同时设置环境变量和context，则只有环境变量中的设置生效|无|可选|无|
|RANK_TABLE_FILE|MindSpore|路径指向文件，包含指定多Ascend AI处理器环境中Ascend AI处理器的"device_id"对应的"device_ip"。|String|文件路径，支持相对路径与绝对路径|与RANK_SIZE配合使用|可选（Ascend AI处理器，使用多卡执行分布式用例时，由用户指定）|无|
|RANK_SIZE|MindSpore|指定深度学习时调用Ascend AI处理器的数量|Integer|1~8，调用Ascend AI处理器的数量|与RANK_TABLE_FILE配合使用|可选（Ascend AI处理器，使用多卡执行分布式用例时，由用户指定）|无|
|RANK_ID|MindSpore|指定深度学习时调用Ascend AI处理器的逻辑ID|Integer|0~7，多机并行时不同server中DEVICE_ID会有重复，使用RANK_ID可以避免这个问题（多机并行时 RANK_ID = SERVER_ID * DEVICE_NUM + DEVICE_ID|无|可选|无|
|MS_RDR_ENABLE|MindSpore|是否开启程序运行数据记录器（RDR），如果MindSpore出现了运行异常，会自动导出MindSpore中预先记录的数据以辅助定位运行异常的原因|Integer|1：开启RDR功能 <br>0：关闭RDR功能|配合`MS_RDR_MODE`与`MS_RDR_PATH`使用|可选|无|
|MS_RDR_MODE|MindSpore|指定运行数据记录器（RDR）导出数据的模式|Integer|1：仅在训练进程异常终止时导出数据 <br>2：训练进程异常终止或正常结束时导出数据|配合`MS_RDR_ENABLE=1`使用|可选|1|
|MS_RDR_PATH|MindSpore|配置程序运行数据记录器（RDR）的文件导出的根目录路径|String|目录路径，仅支持绝对路径|配合`MS_RDR_ENABLE=1`使用，最终RDR文件将保存在`${MS_RDR_PATH}/rank_${RANK_ID}/rdr/`目录下。其中`RANK_ID`为多卡训练场景中的卡号，单卡场景默认`RANK_ID=0`。|可选|无|
|GLOG_v|MindSpore|[日志功能与用法](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_debugging_info.html#id11)|Integer|0-DEBUG <br>1-INFO <br>2-WARNING <br>3-ERROR|无|可选|2|
|GLOG_logtostderr|MindSpore|[日志功能与用法](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_debugging_info.html#id11)|Integer|1:日志输出到屏幕 <br> 0:日志输出到文件|与GLOG_log_dir一起使用|可选|1|
|GLOG_log_dir|MindSpore|[日志功能与用法](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_debugging_info.html#id11)|String|文件路径，支持相对路径与绝对路径|与GLOG_logtostderr一起使用|可选|无|
|GLOG_log_max|MindSpore|[日志功能与用法](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_debugging_info.html#id11)|Integer|正整数|无|可选|50|
|MS_SUBMODULE_LOG_v|MindSpore|[日志功能与用法](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_debugging_info.html#id11)|Dict{String:Integer...}|LogLevel: 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR<br>SubModual: COMMON, MD, DEBUG, DEVICE, COMMON, IR...|无|可选|无|
|GLOG_stderrthreshold|MindSpore|[日志功能与用法](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_debugging_info.html#id11)|Integer|0-DEBUG <br>1-INFO <br>2-WARNING <br>3-ERROR|无|可选|2
|OPTION_PROTO_LIB_PATH|MindSpore|RPOTO依赖库库路径|String|目录路径，支持相对路径与绝对路径|无|可选|无|
|MS_OM_PATH|MindSpore|配置task异常时dump数据路径以及图编译出错时dump的analyze_fail.dat文件的保存目录，保存路径为：指定的路径/rank_${rand_id}/om|String|文件路径，支持相对路径与绝对路径|无|可选|无|
|MINDSPORE_DUMP_CONFIG|MindSpore|指定[云侧Dump功能](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/dump_in_graph_mode.html#id6)或[端侧Dump功能](https://www.mindspore.cn/lite/docs/zh-CN/master/use/benchmark_tool.html#dump)所依赖的配置文件的路径|String|文件路径，支持相对路径与绝对路径|无|可选|无|
|MS_DIAGNOSTIC_DATA_PATH|MindSpore|使用[云侧Dump功能](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/dump_in_graph_mode.html#id6)时，如果Dump配置文件没有设置`path`字段或者设置为空字符串，则“$MS_DIAGNOSTIC_DATA_PATH/debug_dump”就会被当做path的值。若Dump配置文件中设置了`path`字段，则仍以该字段的实际取值为准。|String|文件路径，只支持绝对路径|与MINDSPORE_DUMP_CONFIG配合使用|可选|无|
|MS_ENABLE_CACHE|MindData|是否开启dataset数据处理cache功能，可以实现数据处理过程中数据的cache能力，加速数据集读取及增强处理|String|TRUE：开启数据处理cache功能 <br>FALSE：关闭数据处理cache功能|与MS_CACHE_HOST、MS_CACHE_PORT一起使用|可选|无|
|MS_CACHE_HOST|MindData|开启cache时，cache服务所在的IP|String|Cache Server所在机器的IP|与MS_ENABLE_CACHE=TRUE、MS_CACHE_PORT一起使用|可选|无|
|MS_CACHE_PORT|MindData|开启cache时，cache服务所在的端口|String|Cache Server所在机器的端口|与MS_ENABLE_CACHE=TRUE、MS_CACHE_HOST一起使用|可选|无|
|DATASET_ENABLE_NUMA|MindData|是否开启numa绑核功能，在大多数分布式场景下numa绑核都能提升数据处理效率和端到端性能|String|True: 开启numa绑核功能|与libnuma.so配合使用|可选|无|
|OPTIMIZE|MindData|是否执行dataset数据处理 pipeline 树优化，在适合数据处理算子融合的场景下，可以提升数据处理效率|String|true: 开启pipeline树优化<br>false: 关闭pipeline树优化|无|可选|无|
|ENABLE_MS_DEBUGGER|Debugger|是否在训练中启动Debugger|Boolean|1：开启Debugger<br>0：关闭Debugger|与MS_DEBUGGER_HOST、MS_DEBUGGER_PORT一起使用|可选|无|
|MS_DEBUGGER_HOST|Debugger|MindInsight Debugger服务的IP|String|启动MindInsight调试器的机器的IP|与ENABLE_MS_DEBUGGER=1、MS_DEBUGGER_PORT一起使用|可选|无
|MS_DEBUGGER_PORT|Debugger|连接MindInsight Debugger Server的端口|Integer|1~65536，连接MindInsight Debugger Server的端口|与ENABLE_MS_DEBUGGER=1、MS_DEBUGGER_HOST一起使用|可选|无
|MS_DEBUGGER_PARTIAL_MEM|Debugger|是否开启部分内存复用（只有在Debugger选中的节点才会关闭这些节点的内存复用）|Boolean|1：开启Debugger选中节点的内存复用<br>0：关闭Debugger选中节点的内存复用|无|可选|无|
|GRAPH_OP_RUN|MindSpore|图模式下以任务下沉方式运行pipeline大网络模型时，可能会由于流资源限制而无法正常启动，此环境变量可以指定图模式的执行方式，配置为0表示任务下沉，是默认执行方式；1则表示非任务下沉方式，该方式没有流的限制，但性能有所下降。|Integer|0：执行任务下沉<br>1：执行非任务下沉|无|可选|无|
|GROUP_INFO_FILE|MindSpore|指定通信域信息存储路径|String|通信域信息文件路径，支持相对路径与绝对路径。|无|可选|无|
|MS_DEV_ENABLE_FALLBACK|MindSpore|设置非0值时使能Fallback功能。|Integer|1: 开启Fallback功能<br>0: 关闭Fallback功能 |无|可选|1|