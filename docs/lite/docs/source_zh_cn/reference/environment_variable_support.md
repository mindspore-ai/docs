# 环境变量支持说明

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.2/docs/lite/docs/source_zh_cn/reference/environment_variable_support.md)

本文列举MindSpore Lite所支持的环境变量及其含义说明，并相应地给出了每个环境变量的可选取值和默认取值。

|              **环境变量**              |                                **含义**                                 |         **可选值**          | **默认值**  |
| :------------------------------------: | :---------------------------------------------------------------------: | :-------------------------: | :---------: |
|                 GLOG_v                 |                              日志等级设置                               |           0、1、2、3           |      2      |
|           KEEP_ORIGIN_DTYPE            |                          保持原始数据类型开关                           |            1、""            |     ""      |
|            MSLITE_API_TYPE             |                          benchmark API类型选择                           |           NEW、C            |     NEW     |
|         MINDSPORE_DUMP_CONFIG          |                        端上训练dump数据配置文件                         |        配置文件路径         |     ""      |
|           ASCEND_BACK_POLICY           |                           Ascend后端策略选择                            |          "ge"、""           |     ""      |
|                RANK_ID                 |                                 卡序号                                  |             0-N             |     ""      |
|            ASCEND_DEVICE_ID            |                              Ascend硬件ID                               |           0到7、""           |     ""      |
|             GPU_DEVICE_ID              |                                GPU硬件ID                                |           0到7、""           |     ""      |
|      BENCHMARK_UPDATE_CONFIG_ENV       |                         Benchmark工具配置项设置                         |           "0"、""            |     ""      |
|          MSLITE_PACKAGE_PATH           |                            测试用例打包路径                             |          文件路径           |     ""      |
|     MS_ASCEND_CHECK_OVERFLOW_MODE      |                              精度模式选择                               | SATURATION_MODE/INFNAN_MODE | INFNAN_MODE |
|          disable_REUSE_MEMORY          |                        Ascend GE后端显存复用开关                        |           "0"、"1"           |      0      |
|      ENABLE_MULTI_BACKEND_RUNTIME      |                           多后端异构能力开关                            |         "on"、"off"          |     off     |
|         ASCEND_CUSTOM_OPP_PATH         |                        Ascend C自定义算子安装路径                        |          文件路径           |     ""      |
|            ASCEND_OPP_PATH             |                             Ascend算子路径                              |          文件路径           |     ""      |
|     MSLITE_ENABLE_CLOUD_INFERENCE      |                            是否使能云侧推理                             |          "on"、""           |     ""      |
|               ENABLE_AKG               |                               是否使能AKG                               |          "on"、""           |     ""      |
|         MS_INDEPENDENT_DATASET         |                           是否使用外部数据集                            |         "true"、""          |     ""      |
|                OPTIMIZE                |                     MindData业务场景下是否使能优化                      |         "true"、""          |     ""      |
|             MS_CACHE_HOST              |                      MindData业务场景下的主机地址                       |          主机地址           |  127.0.0.1  |
|             MS_CACHE_PORT              |                      MindData业务场景下的主机端口                       |           端口号            |    50052    |
|               DEVICE_ID                |                    端上MindData使用场景下设置硬件ID                     |           0到7、""           |     ""      |
|             MS_CPU_FEATURE             |                               CPU指令架构                               |           avx512            |     ""      |
| MS_DEV_GRAPH_KERNEL_SPLIT_DEBUG_TUNING |                              切图调试开关                               |           on、""            |     ""      |
|      MS_DEV_DUMP_GRAPH_KERNEL_IR       |                                 Dump IR                                 |           on、""            |     ""      |
|               TIME_STEP                |                                迭代次数                                 |        整数类型数据         |     ""      |
|              MAX_ROI_NUM               | 若用户模型含有proposal算子，需根据proposal算子实现情况配置MAX_ROI_NUM |        整数类型数据         |     300     |
|            PARA_GROUP_FILE             |                             通信域配置文件                              |        配置文件路径         |     ""      |
|             MS_ENABLE_HCCL             |                              是否使能HCCL                               |        0（空）、非0         |   0（空）   |

