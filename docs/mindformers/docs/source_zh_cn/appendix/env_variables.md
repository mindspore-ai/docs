# 环境变量说明

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/appendix/env_variables.md)

本文介绍MindFormers的环境变量。

## 调试变量

| 变量名称                        | 默认值 | 解释                                                                                        | 说明                                                                                                                                                                     |
| ------------------------------- | ------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **HCCL_DETERMINISTIC**          | false  | 是否打开HCCL通信库的确定性开关，消除AllReduce等多卡求和顺序不一致引入的随机性。（影响性能） | `true`：打开HCCL确定性开关；<br>`false`：关闭HCCL确定性开关。                                                                                                                    |
| **LCCL_DETERMINISTIC**          | 0      | 是否打开LCCL通信库的确定性行为。                                                            | `1`：打开LCCL确定性开关；<br>`0`：关闭LCCL确定性开关。                                                                                                                         |
| **CUSTOM_MATMUL_SHUFFLE**       | on     | 是否开启自定义矩阵乘法的洗牌操作。                                                          | `on`：开启矩阵洗牌；<br>`off`：关闭矩阵洗牌。                                                                                                                                |
| **ASCEND_LAUNCH_BLOCKING**      | 0      | 是否等待所有进程就绪后再启动进程。从而控制推理精度一致性。                                  | `1`：等待所有进程就绪后再启动进程；<br>`0`：不等待所有进程就绪就启动进程。                                                                                                   |
| **TE_PARALLEL_COMPILER**        | 8      | 算子并行编译的线程数。                                                                      | 取值为正整数；设置为`1`时为单线程编译，简化问题调试难度。                                                                                                                  |
| **CPU_AFFINITY**                | 0      | 是否启动CPU亲和性开关，从而确保每个进程或线程绑定到一个CPU核心上，以提高性能。              | `1`：开启CPU亲和性开关；<br>`0`：关闭CPU亲和性开关。                                                                                                                         |
| **LOG_MF_PATH**                 | NA     | MindFormers日志保存位置。                                                                     | 文件路径，支持相对路径与绝对路径。                                                                                                                                             |
| **MS_MEMORY_STATISTIC**         | 0      | 内存析构。                                                                                    | `1`：开启内存析构功能；<br>`0`：关闭内存析构功能。                                                                                                                             |
| **MINDSPORE_DUMP_CONFIG**       | NA      | 指定Dump功能所依赖的配置文件的路径。                                                          | 文件路径，支持相对路径与绝对路径。                                                                                                                                       |
| **GLOG_v**                      | 2      | 控制MindSpore日志的级别。                                                                     | `0`：DEBUG；<br>`1`：INFO；<br>`2`：WARNING；<br>`3`：ERROR：表示程序执行出现报错，输出错误日志，程序可能不会终止；<br>`4`：CRITICAL，表示程序执行出现异常，将会终止执行程序。 |
| **ASCEND_GLOBAL_LOG_LEVEL**     | 3      | 控制CANN的日志级别。                                                                          | `0`：DEBUG；<br>`1`：INFO；<br>`2`：WARNING；<br>`3`：ERROR；<br>`4`：CRITICAL。                                                                                               |
| **ASCEND_SLOG_PRINT_TO_STDOUT** | 0      | 是否打开plog日志打屏开关。                                                                    | `1`：开启日志打屏；<br>`0`：关闭日志打屏。                                                                                                                                   |
| **ASCEND_GLOBAL_EVENT_ENABLE**  | 0      | 是否开启event事件日志。                                                                       | `1`：开启Event日志；<br>`0`：关闭Event日志。                                                                                                                                 |
| **HCCL_EXEC_TIMEOUT**           | 1800   | HCCL进程执行同步等待时间。                                                                    | 执行同步等待时间（s）。                                                                                                                                                    |
| **HCCL_CONNECT_TIMEOUT**        | 120    | HCCL建链超时等待时间。                                                                        | 建链等待时间（s）。                                                                                                                                                        |

## 其它变量

| 变量名称                           | 默认值      | 解释                                      | 说明                                                                                                                                                      |
| ---------------------------------- | ----------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RUN_MODE**                       | predict     | 设置运行模式。                            | `predict`：推理； <br>`finetune`：微调； <br>`train`：训练； <br>`eval`：评测。                                                                                            |
| **USE_ROPE_SELF_DEFINE**           | true        | 是否启用自定义的ROPE。                    | `true`：启用自定义的ROPE；<br>`false`：关闭自定义的ROPE。                                                                                                     |
| **MS_ENABLE_INTERNAL_BOOST**       | off         | 是否打开MindSpore框架的内部加速功能。     | `on`：开启MindSpore内部加速；<br>`off`：关闭MindSpore内部加速。                                                                                               |
| **MS_GE_ATOMIC_CLEAN_POLICY**      | 1           | 是否集中清理网络中atomic算子占用的内存。  | `0`：集中清理网络中所有atomic算子占用的内存；<br>`1`：不集中清理内存，对网络中每一个atomic算子进行单独清零。                                                  |
| **ENABLE_LAZY_INLINE**             | 1           | 是否开启lazy inline。                     | `0`：关闭lazy inline；<br>`1`：开启lazy inline。                                                                                                              |
| **ENABLE_LAZY_INLINE_NO_PIPELINE** | 0           | 是否开启在非pipeline并行下的lazy inline。 | `0`：关闭lazy inline；<br>`1`：关闭lazy inline。                                                                                                              |
| **MS_ASCEND_CHECK_OVERFLOW_MODE**  | INFNAN_MODE | 设置溢出检测模式。                        | `SATURATION_MODE`：饱和模式，计算出现溢出时，饱和为浮点数极值（+-MAX）；<br>`INFNAN_MODE`：INF/NAN模式，遵循IEEE 754标准，根据定义输出INF/NAN的计算结果。 |

MindSpore相关环境变量请参考以下链接：

[MindSpore环境变量](https://www.mindspore.cn/docs/zh-CN/master/note/env_var_list.html)
