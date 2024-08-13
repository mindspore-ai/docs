# 功能调试

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/debug.md)

## 调优常见问题及解决办法

- 显存调试阶段，可能遇到以下常见问题：
    - Malloc device memory failed:
         MindSpore申请device侧内存失败，原始是设备被其他进程占用，可通过ps -ef | grep "python"查看正在跑的进程。
    - Out of Memory：
         MindSpore申请动态内存失败，可能的原因有：batch size太大，处理数据太多导致内存占用大；通信算子占用内存太多导致整体内存复用率较低。

## MindSpore功能调试介绍

在网络的迁移过程，建议优先使用PYNATIVE模式进行调试，在PYNATIVE模式下可以进行debug，日志打印也比较友好。在调试ok后转成图模式运行，图模式在执行性能上会更友好，也可以找到一些在编写网络中的问题，比如使用了三方的算子导致梯度截断。
详情请参考[错误分析](https://www.mindspore.cn/docs/zh-CN/master/model_train/debug/error_analysis/error_scenario_analysis.html)。
