# 模型相关

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_zh_cn/faq/model_related.md)

## Q: 网络运行时报错“Out of Memory”(`OOM`)，如何处理？

A: 首先上述报错指的是设备内存不足，导致这一报错的原因可能有多种，建议进行如下几方面的排查:

1. 使用命令`npu-smi info`，确认卡是否独占状态。
2. 建议运行网络时，使用对应网络默认`yaml`配置。
3. 网络对应`yaml`配置文件中适当增大`max_device_memory`的值，注意需要给卡间通信预留部分内存，可以渐进性增大进行尝试。
4. 调整混合并行策略，适当增大流水线并行（pp）和模型并行（mp），并相应减小数据并行（dp），保持`dp * mp * pp = device_num`，有必要时增加NPU数量。
5. 尝试调小批次大小或序列长度。
6. 打开选择重计算或完全重计算，打开优化器并行。
7. 如问题仍需进一步排查，欢迎[提issue](https://gitee.com/mindspore/mindformers/issues)反馈。

<br/>