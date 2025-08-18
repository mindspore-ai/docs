# 环境变量清单

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.0/docs/vllm_mindspore/docs/source_zh_cn/user_guide/environment_variables/environment_variables.md)

|   环境变量   |   功能   |   类型   |   取值   |   说明   |
|   ------   |   -------  |   ------   |   ------   |   ------   |
|   `vLLM_MODEL_BACKEND`   |   用于指定模型后端。使用vLLM MindSpore原生模型后端时无需指定；使用模型为vLLM MindSpore外部后端时则需要指定。   |   String   | `MindFormers`: 模型后端为MindSpore Transformers。   |   原生模型后端当前支持Qwen2.5系列；MindSpore Transformers模型后端支持Qwen系列、DeepSeek、Llama系列模型，使用时需配置环境变量：`export PYTHONPATH=/path/to/mindformers/:$PYTHONPATH`。   |
|   `MINDFORMERS_MODEL_CONFIG`   |   MindSpore Transformers模型的配置文件。使用Qwen2.5系列、DeepSeek系列模型时，需要配置文件路径。   |   String   |   模型配置文件路径。   |   **该环境变量在后续版本会被移除。** 样例：`export MINDFORMERS_MODEL_CONFIG=/path/to/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b_w8a8.yaml`。   |
|   `GLOO_SOCKET_IFNAME`   |   用于多机之间使用gloo通信时的网口名称。   |   String   |  网口名称，例如enp189s0f0。    |   多机场景使用，可通过`ifconfig`查找ip对应网卡的网卡名。   |
|   `TP_SOCKET_IFNAME`   |   用于多机之间使用TP通信时的网口名称。   |   String   | 网口名称，例如enp189s0f0。      |   多机场景使用，可通过`ifconfig`查找ip对应网卡的网卡名。   |
| `HCCL_SOCKET_IFNAME` | 用于多机之间使用HCCL通信时的网口名称。 | String | 网口名称，例如enp189s0f0。  | 多机场景使用，可通过`ifconfig`查找ip对应网卡的网卡名。 |
| `ASCEND_RT_VISIBLE_DEVICES` | 指定哪些Device对当前进程可见，支持一次指定一个或多个Device ID。 | String | 为Device ID，逗号分割的字符串，例如"0,1,2,3,4,5,6,7"。 | ray使用场景建议使用。 |
| `HCCL_BUFFSIZE` | 此环境变量用于控制两个NPU之间共享数据的缓存区大小。 | Integer | 缓存区大小，大小为MB。例如：`2048`。 | 使用方法参考：[HCCL_BUFFSIZE](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1beta1/maintenref/envvar/envref_07_0080.html)。例如DeepSeek 混合并行（数据并行数为32，专家并行数为32），且`max-num-batched-tokens`为256时，则`export HCCL_BUFFSIZE=2048`。 |
| `MS_MEMPOOL_BLOCK_SIZE` | 设置PyNative模式下设备内存池的块大小。 | String | 正整数string，单位为GB。 |  |
| `vLLM_USE_NPU_ADV_STEP_FLASH_OP` | 是否使用昇腾`adv_step_flash`算子。 | String | `on`: 使用；`off`：不使用 | 取值为`off`时，将使用小算子实现替代`adv_step_flash`算子。 |
| `VLLM_TORCH_PROFILER_DIR` | 开启profiling采集数据，当配置了采集数据保存路径后生效 | String | Profiling数据保存路径。|   |

更多的环境变量信息，请查看：

- [CANN 环境变量列表](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1beta1/index/index.html)
- [MindSpore 环境变量列表](https://www.mindspore.cn/docs/zh-CN/r2.7.0/api_python/env_var_list.html)
- [MindSpore Transformers 环境变量列表](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.6.0/index.html)
- [vLLM 环境变量列表](https://docs.vllm.ai/en/v0.8.4/serving/env_vars.html)
