# 功能相关

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.10/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.10/docs/mindformers/docs/source_zh_cn/faq/func_related.md)

## Q: 如何生成模型切分策略文件？

A: 模型切分策略文件记录了模型权重在分布式场景下的切分策略，一般在离线权重切分时使用。在网络`yaml`文件中配置`only_save_strategy: True`，然后正常启动分布式任务，便可在`output/strategy/`目录下生成分布式策略文件，详细介绍请参阅[分布式权重切分与合并教程](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/function/transform_weight.html#%E7%A6%BB%E7%BA%BF%E8%BD%AC%E6%8D%A2%E9%85%8D%E7%BD%AE%E8%AF%B4%E6%98%8E)。

<br/>

## Q: 生成`ranktable`文件报错`socket.gaierror: [Errno -2] Name or service not known`或者`socket.gaierror: [Errno -3] Temporary failure in name resolution`，怎么解决？

A: 从`MindSpore Transformers r1.2.0`版本开始，集群启动统一使用`msrun`方式，`ranktable`启动方式已废弃。

<br/>
