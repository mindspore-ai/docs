# Memory Reuse

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/optimize/mem_reuse.md)

## Overview

The memory reuse is to let different Tensors share the same part of the memory to reduce memory overhead and support a larger network. After shutting down, each Tensor has its own independent memory space, and tensors have no shared memory.
The MindSpore memory reuse function is turned on by default, and the function can be manually controlled to turn off and on in the following ways.

## Usage

1. Construct configuration file `mindspore_config.json`.

    ```json
    {
        "sys": {
            "mem_reuse": true
        }
    }
    ```

    > mem_reuse: controls whether the memory reuse function is turned on. When it is set to true, the control memory reuse function is turned on, and when false, the memory reuse function is turned off.

2. Configure the memory reuse function through `context`.

    ```python
    import mindspore as ms
    ms.set_context(env_config_path="./mindspore_config.json")
    ```