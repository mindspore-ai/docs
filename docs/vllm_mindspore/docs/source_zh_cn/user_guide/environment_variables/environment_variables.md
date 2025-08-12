# 环境变量清单

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.0/docs/vllm_mindspore/docs/source_zh_cn/user_guide/environment_variables/environment_variables.md)

|   环境变量   |   功能   |   类型   |   取值   |   说明   |
|   ------   |   -------  |   ------   |   ------   |   ------   |
|   vLLM_MODEL_BACKEND   |   用于指定模型后端。使用vLLM MindSpore原生模型后端时无需指定；使用模型为vLLM MindSpore外部后端时则需要指定。   |   String   | `MindFormers`: 模型后端为MindSpore Transformers。   |   原生模型后端当前支持Qwen2.5系列；MindSpore Transformers模型后端支持Qwen系列、DeepSeek、Llama系列模型，使用时需配置环境变量：`export PYTHONPATH=/path/to/mindformers/:$PYTHONPATH`。   |
|   MINDFORMERS_MODEL_CONFIG   |   MindSpore Transformers模型的配置文件。使用Qwen2.5系列、DeepSeek系列模型时，需要配置文件路径。   |   String   |   模型配置文件路径。   |   **该环境变量在后续版本会被移除。** 样例：`export MINDFORMERS_MODEL_CONFIG=/path/to/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b_w8a8.yaml`。   |
|   GLOO_SOCKET_IFNAME   |   用于多机之间使用gloo通信时的网口名称。   |   String   |  网口名称，例如enp189s0f0。    |   多机场景使用，可通过`ifconfig`查找ip对应网卡的网卡名。   |
|   TP_SOCKET_IFNAME   |   用于多机之间使用TP通信时的网口名称。   |   String   | 网口名称，例如enp189s0f0。      |   多机场景使用，可通过`ifconfig`查找ip对应网卡的网卡名。   |
| HCCL_SOCKET_IFNAME | 用于多机之间使用HCCL通信时的网口名称。 | String | 网口名称，例如enp189s0f0。  | 多机场景使用，可通过`ifconfig`查找ip对应网卡的网卡名。 |
| ASCEND_RT_VISIBLE_DEVICES | 指定哪些Device对当前进程可见，支持一次指定一个或多个Device ID。 | String | 为Device ID，逗号分割的字符串，例如"0,1,2,3,4,5,6,7"。 | ray使用场景建议使用。 |
| HCCL_BUFFSIZE | 此环境变量用于控制两个NPU之间共享数据的缓存区大小。 | Integer | 缓存区大小，大小为MB。例如：`2048`。 | 使用方法参考：[HCCL_BUFFSIZE](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1beta1/maintenref/envvar/envref_07_0080.html)。例如DeepSeek 混合并行（数据并行数为32，专家并行数为32），且`max-num-batched-tokens`为256时，则`export HCCL_BUFFSIZE=2048`。 |
| MS_MEMPOOL_BLOCK_SIZE | 设置PyNative模式下设备内存池的块大小。 | String | 正整数string，单位为GB。 |  |
| vLLM_USE_NPU_ADV_STEP_FLASH_OP | 是否使用昇腾`adv_step_flash`算子。 | String | `on`: 使用；`off`：不使用 | 取值为`off`时，将使用小算子实现替代`adv_step_flash`算子。 |
| VLLM_TORCH_PROFILER_DIR | 开启profiling采集数据，当配置了采集数据保存路径后生效 | String | Profiling数据保存路径。|   |

以下环境变量由vLLM MindSpore自动注册：

|   环境变量   |   功能   |   类型   |   取值   |   说明   |
|   ------   |   -------  |   ------   |   ------   |   ------   |
|   USE_TORCH   |   Transformer运行时依赖该环境变量  |   String   |   默认值为"False"   | vLLM MindSpore 不使用torch 作为后端   |
|   USE_TF   |   Transformer运行时依赖该环境变量  |   String   |   默认值为"False"   | vLLM MindSpore 不使用TensorFlow 作为后端   |
|   RUN_MODE   |   执行模式为推理  |   String   |   默认值为"predict"   |  **该环境变量在后续版本会被移除。** 为MindFormers依赖的环境变量   |
|   CUSTOM_MATMUL_SHUFFLE   |   开启或关闭自定义矩阵算法的洗牌操作  |   String   |   `on`：开启矩阵洗牌。`off`：关闭矩阵洗牌。默认值为`on`。   | |
|   HCCL_DETERMINISTIC   |   开启或关闭归约类通信算子的确定性计算，其中归约类通信算子包括 AllReduce、ReduceScatter、Reduce。  |   String   |   `true`：打开 HCCL 确定性开关；`false`：关闭 HCCL 确定性开关。默认值为`false`。   |    |
|   ASCEND_LAUNCH_BLOCKING   |   训练或在线推理场景，可通过此环境变量控制算子执行时是否启动同步模式。  |   Integer   |   `1`：强制算子采用同步模式运行；`0`：不强制算子采用同步模式运行。默认值为`0`。   |     |
|   TE_PARALLEL_COMPILER   |   算子最大并行编译进程数，当大于 1 时开启并行编译。  |   Integer   |   取值为正整数；最大不超过 cpu 核数*80%/昇腾 AI 处理器个数，取值范围 1~32。默认值是 `0`。   |     |
|   LCCL_DETERMINISTIC   |   设置 LCCL 确定性算子 AllReduce(保序加)是否开启。  |   Integer   |   `1`：打开 LCCL 确定性开关；`0`：关闭 LCCL 确定性开关。默认值是 `0`。   |    |
|   MS_ENABLE_GRACEFUL_EXIT   |   设置使能进程优雅退出  |   Integer   |   `1`：使用进程优雅退出功能。`不设置或者其他值`: 不使用进程优雅退出功能。默认值为`0`   |      |
|   CPU_AFFINIITY   |   MindSpore推理绑核优化  |   String   |   `True`：开启绑核；`True`：不开启绑核。默认值为`True`   |   **该环境变量在后续版本会被移除。** 将使用`set_cpu_affinity`接口。   |
|   MS_ENABLE_INTERNAL_BOOST   |   是否打开 MindSpore 框架的内部加速功能。  |   String   |   `on`：开启 MindSpore 内部加速；`off`：关闭 MindSpore 内部加速。默认值为`on`   |    |
|   MS_ENABLE_LCCL   |   是否使用LCCL通信库。  |   Integer   |   `1`:开启，`0`:关闭。默认值为`0`。   |     |
|   HCCL_EXEC_TIMEOUT   |   通过该环境变量可控制设备间执行时同步等待的时间，在该配置时间内各设备进程等待其他设备执行通信同步。  |   Integer   |   取值范围为：(0, 17340]，单位为 s。 默认值为 7200。  |    |
|   DEVICE_NUM_PER_NODE   | 节点上的设备数    |   Integer   |   默认值为16。   |   |
|   HCCL_OP_EXPANSION_MODE   |   用于配置通信算法的编排展开位置。  |   String   |  `AI_CPU`：通信算法的编排展开位置为Device侧的AI CPU计算单元；`AIV`：通信算法的编排展开位置为Device侧的AI Vector Core计算单元。默认值为`AIV`。   |     |
|   MS_JIT_MODULES   |   指定静态图模式下哪些模块需要JIT静态编译，其函数方法会被编译成静态计算图  |   String   |   模块名，对应import导入的顶层模块的名称。如果有多个，使用英文逗号分隔。默认值为`"vllm_mindspore,research"`。   |     |
|   GLOG_v   |   控制日志的级别  |   Integer   |   `0`：DEBUG；`1`：INFO；`2`：WARNING；`3`：ERROR，表示程序执行出现报错，输出错误日志，程序可能不会终止；`4`：CRITICAL，表示程序执行出现异常，将会终止执行程序。默认值为`3`。   |      |
|   RAY_CGRAPH_get_timeout   |   `ray.get()`方法的超时时间。  |   Integer   |   默认值为`360`。   |      |
|   MS_NODE_TIMEOUT |   节点心跳超时时间，单位：秒。     |   Integer  |   默认值为`180`。   |    |

更多的环境变量信息，请查看：
 - [CANN 环境变量列表](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1beta1/index/index.html)
 - [MindSpore 环境变量列表](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/env_var_list.html)
 - [MindSpore Transformers 环境变量列表](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.6.0/index.html)
 - [vLLM 环境变量列表](https://docs.vllm.ai/en/v0.8.4/serving/env_vars.html)
