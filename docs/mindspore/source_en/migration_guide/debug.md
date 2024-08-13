# Function Debugging

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/migration_guide/debug.md)

## Debugging Tools

- The following common problems may be encountered during the graphics debugging phase:
    - Malloc device memory failed:
         MindSpore failed to request memory on the device side, the original memory is that the device is occupied by other processes, you can check the running processes by ps -ef | grep "python".
    - Out of Memory:
         The possible reasons for failure to request dynamic memory are: batch size is too large, processing too much data leads to a large memory footprint; communication operators take up too much memory leading to a low overall memory reuse rate.

## Introduction of MindSpore Debugging

### Function Debugging

During network migration, you are advised to use the PyNative mode for debugging. In PyNative mode, you can perform debugging, and log printing is user-friendly. After the debugging is complete, the graph mode is used. The graph mode is more user-friendly in execution performance. You can also find some problems in network compilation. For example, gradient truncation caused by third-party operators.
For details, see [Error Analysis](https://www.mindspore.cn/docs/en/master/model_train/debug/error_analysis/error_scenario_analysis.html).
