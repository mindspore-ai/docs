# 环境变量说明

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/appendix/env_variables.md)

本文介绍MindFormers的环境变量。

## 调试变量

| 变量名称                   | 默认值 | 解释                                                           | 说明                                                   |
| -------------------------- | ------ | -------------------------------------------------------------- | ------------------------------------------------------ |
| **HCCL_DETERMINISTIC**     | false   | 是否打开HCCL通信库的确定性开关，消除AllReduce等多卡求和顺序不一致引入的随机性。（影响性能）                               | true：打开HCCL确定性开关；false：关闭HCCL确定性开关。            |
| **LCCL_DETERMINISTIC**     | 0      | 是否打开LCCL通信库的确定性行为。                               | 1：打开LCCL确定性开关；0：关闭LCCL确定性开关                      |
| **CUSTOM_MATMUL_SHUFFLE**  | on     | 是否开启自定义矩阵乘法的洗牌操作。                             | on：开启矩阵洗牌；off：关闭矩阵洗牌。                  |
| **ASCEND_LAUNCH_BLOCKING** | 0      | 是否等待所有进程就绪后再启动进程。从而控制推理精度一致性。 | 1：等待所有进程就绪后再启动进程；0：不等待所有进程就绪就启动进程。 |
| **TE_PARALLEL_COMPILER**     | 8      | 算子并行编译的线程数。  | 取值为正整数；设置为1时为单线程编译，简化问题调试难度。                  |
| **CPU_AFFINITY**             | 0      | 是否启动CPU亲和性开关，从而确保每个进程或线程绑定到一个CPU核心上，以提高性能。 | 1：开启CPU亲和性开关；0：关闭CPU亲和性开关。|

## 其它变量

| 变量名称                 | 默认值  | 解释                     | 说明                                                      |
| ------------------------ | ------- | ---------------------- | --------------------------------------------------------- |
| **RUN_MODE**             | predict | 设置运行模式。          | 可选的模式包括 `predict`, `finetune`, `train`, `eval` |
| **USE_ROPE_SELF_DEFINE** | true    | 是否启用自定义的ROPE。  |  true：启用自定义的ROPE；false：关闭自定义的ROPE。  |
| **MS_ENABLE_INTERNAL_BOOST** | off    | 是否打开MindSpore框架的内部加速功能。 | on：开启MindSpore内部加速；off：关闭MindSpore内部加速。 |
