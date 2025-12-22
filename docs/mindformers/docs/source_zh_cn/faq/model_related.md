# 模型相关 FAQ

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.2/docs/mindformers/docs/source_zh_cn/faq/model_related.md)

## Q: 网络运行时报错“Out of Memory”(`OOM`)，如何处理？

A: 该报错表示设备内存不足，可能由多种原因导致，建议按以下方面排查：

1. 使用命令`npu-smi info`，确认卡是否独占状态。
2. 建议运行网络时，使用对应网络默认`yaml`配置。
3. 在对应网络的`yaml`配置文件中适当增大`max_device_memory`的值。注意需要给卡间通信预留部分内存，可以渐进性增大进行尝试。
4. 调整混合并行策略，适当增大流水线并行（pp）和模型并行（mp），并相应减小数据并行（dp），保持`dp * mp * pp = device_num`，必要时增加NPU数量。
5. 尝试调小批次大小或序列长度。
6. 开启选择重计算或完全重计算，开启优化器并行。
7. 如问题仍需进一步排查，欢迎[提issue](https://gitee.com/mindspore/mindformers/issues)反馈。
