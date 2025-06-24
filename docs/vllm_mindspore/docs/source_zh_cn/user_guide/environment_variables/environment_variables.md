# 环境变量清单

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/user_guide/environment_variables/environment_variables.md)

|   环境变量   |   功能   |   类型   |   取值   |   说明   |
|   ------   |   -------  |   ------   |   ------   |   ------   |
|   vLLM_MODEL_BACKEND   |   用于指定模型来源。当使用的模型为vLLM MindSpore外部模型时则需要指定。   |   String   |   MindFormers:  模型来源为MindSpore Transformers   |   当模型来源为MindSpore Transformers，使用Qwen2.5系列、DeepSeek系列模型时，需要配置环境变量：`export PYTHONPATH=/path/to/mindformers/:$PYTHONPATH`   |
|   MINDFORMERS_MODEL_CONFIG   |   MindSpore Transformers模型的配置文件。使用Qwen2.5系列、DeepSeek系列模型时，需要配置文件路径。   |   String   |   模型配置文件路径   |   **该环境变量在后续版本会被移除。**样例：`export MINDFORMERS_MODEL_CONFIG=/path/to/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b_w8a8.yaml`   |
|   GLOO_SOCKET_IFNAME   |   用于多机之间使用gloo通信时的网口名称。   |   String   |  网口名称，例如enp189s0f0    |   多机场景使用，可通过`ifconfig`查找ip对应网卡的网卡名。   |
|   TP_SOCKET_IFNAME   |   用于多机之间使用TP通信时的网口名称。   |   String   | 网口名称，例如enp189s0f0      |   多机场景使用，可通过`ifconfig`查找ip对应网卡的网卡名。   |
| HCCL_SOCKET_IFNAME | 用于多机之间使用HCCL通信时的网口名称。 | String | 网口名称，例如enp189s0f0  | 多机场景使用，可通过`ifconfig`查找ip对应网卡的网卡名。 |
| ASCEND_RT_VISIBLE_DEVICES | 指定哪些Device对当前进程可见，支持一次指定一个或多个Device ID。 | String | 为Device ID，逗号分割的字符串，例如"0,1,2,3,4,5,6,7" | ray使用场景建议使用 |
| HCCL_BUFFSIZE | 此环境变量用于控制两个NPU之间共享数据的缓存区大小。 | int | 缓存区大小，大小为MB。例如：`2048` | 使用方法参考：[HCCL_BUFFSIZE](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1beta1/maintenref/envvar/envref_07_0080.html)。例如DeepSeek 混合并行（数据并行数为32，专家并行数为32），且`max-num-batched-tokens`为256时，则`export HCCL_BUFFSIZE=2048` |
