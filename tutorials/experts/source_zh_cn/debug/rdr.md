# Running Data Recorder

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.1/tutorials/experts/source_zh_cn/debug/rdr.md)

## 概述

Running Data Recorder(RDR)是MindSpore提供训练程序运行时记录数据的功能。要记录的数据将会在MindSpore中进行预设，运行训练脚本时，如果MindSpore出现了运行异常，则会自动地导出MindSpore中预先记录的数据以辅助定位运行异常的原因。不同的运行异常将会导出不同的数据，比如出现`Run task error`异常，将会导出计算图、图执行顺序、内存分配等信息以辅助定位异常的原因。

> 并非所有运行异常都会导出数据，目前仅支持部分异常导出数据。
>
> 目前仅支持图模式训练场景下，收集CPU/Ascend/GPU的相关数据。

## 使用方法

### 通过配置文件配置RDR

1. 创建配置文件`mindspore_config.json`。

    ```json
    {
        "rdr": {
            "enable": true,
            "mode": 1,
            "path": "/path/to/rdr/dir"
        }
    }
    ```

    > enable：控制RDR功能是否开启。
    >
    > mode：控制RDR数据导出模式，设置为1表示仅在训练异常终止时导出数据，设置为2表示训练异常终止或正常结束时导出数据。
    >
    > path：设置RDR保存数据的路径，仅支持绝对路径。

2. 通过 `context` 配置RDR。

    ```python
    import mindspore as ms
    ms.set_context(env_config_path="./mindspore_config.json")
    ```

### 通过环境变量配置RDR

通过`export MS_RDR_ENABLE=1`来开启RDR，通过`export MS_RDR_MODE=1`或`export MS_RDR_MODE=2`来设置导出数据模式，然后通过`export MS_RDR_PATH=/path/to/root/dir`设置RDR文件导出的根目录路径，最终RDR文件将保存在`/path/to/root/dir/rank_{RANK_ID}/rdr/`目录下。其中`RANK_ID`为多卡训练场景中的卡号，单卡场景默认`RANK_ID=0`。

> 用户设置的配置文件优先级高于环境变量。

### 异常处理

假如在Ascend 910上使用MindSpore进行训练，训练出现了`Run task error`异常。

这时我们到RDR文件的导出目录中，可以看到有几个文件，每一个文件都代表着一种数据。比如 `hwopt_d_before_graph_0.ir` 该文件为计算图文件。可以使用文本工具打开该文件，用以查看计算图，分析计算图是否符合预期。

### 诊断处理

当开启RDR并设置环境变量`export MS_RDR_MODE=2`，进入诊断模式。在图编译结束后，我们同样可以在RDR文件的导出目录中看到保存的与异常处理相同的文件。
