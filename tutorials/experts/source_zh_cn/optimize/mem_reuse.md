# 内存复用

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.1/tutorials/experts/source_zh_cn/optimize/mem_reuse.md)

## 概述

内存复用功能(Mem Reuse)是让不同的Tensor共用同样的一部分内存，以降低内存开销，支撑更大的网络，关闭后每个Tensor有自己独立的内存空间，Tensor间无共享内存。
MindSpore内存复用功能默认开启，可以通过以下方式手动控制该功能的关闭和开启。

## 使用方法

1. 创建配置文件`mindspore_config.json`。

    ```json
    {
        "sys": {
            "mem_reuse": true
        }
    }
    ```

    > mem_reuse: 控制内存复用功能是否开启，当设置为true时，控制内存复用功能开启，为false时，内存复用功能关闭。

2. 通过 `context` 配置内存复用功能。

    ```python
    import mindspore as ms
    ms.set_context(env_config_path="./mindspore_config.json")
    ```
