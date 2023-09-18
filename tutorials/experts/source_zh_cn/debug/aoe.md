# AOE调优工具

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/debug/aoe.md)&nbsp;&nbsp;

## 概述

AOE（Ascend Optimization Engine）是一款自动调优工具，作用是充分利用有限的硬件资源，以满足算子和整网的性能要求。AOE工具的详细介绍，请参考[AOE简介](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha001/developmenttools/devtool/aoe_16_001.html)。本文档主要介绍如何使用AOE工具实现MindSpore训练场景下的调优。

## 开启调优

1. 在线调优

    在set_context接口中设置`aoe_tune_mode`，即可开启AOE工具进行在线调优。`aoe_tune_mode`的取值应当在`["online", "offline"]`中。其中：

    online：开启在线调优。

    offline：为离线调优保存GE图。当通过`set_context(save_graphs=True, save_graphs_path="path/to/ir/files")`设置了保存图的路径，图保存在指定路径的aoe_dump目录下；否则保存在当前运行目录下面的aoe_dump下。

    在set_context接口中设置`aoe_config`，可设置调优配置。`job_type`是设置调优类型，取值在`["1", "2"]`中，默认值是`2`。其中：

    1：表示子图调优。

    2：表示算子调优。

    举例在线调优的使用方法：

    ```python
    import mindspore as ms
    ms.set_context(aoe_tune_mode="online", aoe_config={"job_type": "2"})
    ....
    ```

    设置好上述context之后，按照正常执行训练脚本方式即可启动调优，用例执行期间，无需任何操作，用例执行结束之后的结果即为调优之后的结果。

2. 离线调优

    离线调优则是使用训练脚本生成网络模型时的Dump数据（包含算子输出描述文件、算子的二进制文件等）进行算子调优。离线调优的启动方式以及相关环境变量可参考`CANN`开发工具指南的[离线调优](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha001/developmenttools/devtool/aoe_16_023.html)。

## 查看调优结果

调优开始后，会在执行调优的工作目录下生成命名为`aoe_result_opat_{timestamp}_{pidxxx}.json`的文件来记录调优过程和调优结果。该文件的具体解析请参考[调优结果文件分析](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha001/developmenttools/devtool/aoe_16_028.html)。

调优完成后，若满足自定义知识库生成条件，则会生成自定义知识库。如果指定了知识库存储路径的环境变量`TUNE_BANK_PATH`，调优生成的知识库会在指定目录下生成，否则调优生成的知识库会在如下默认路径中`${HOME}/Ascend/latest/data/aoe/custom/graph/${soc_version}`。

## 知识库合并

算子调优结束后，生成的调优知识库支持合并以便于再次执行用例使用（或者其他脚本使用）。仅支持相同昇腾AI处理器型号下的自定义知识库合并。具体合并方式请参考`CANN`开发工具指南中的[合并知识库](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha001/developmenttools/devtool/aoepar_16_061.html)。

## 使用须知

AOE调优工具在使用时，请注意以下几点：

1. AOE调优工具只支持在`Ascend`环境上使用。

2. 请确保运行环境中执行调优用户的home目录下磁盘可用空间>=20G。

3. AOE调优工具依赖部分第三方软件`pciutils`。

4. 开启该调优工具后，可以明显感知算子编译时间变长，属于正常现象。