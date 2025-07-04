# 功能相关 FAQ

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/faq/feature_related.md)

## Q: WikiText数据集下载链接失效。

A: 官方下载链接失效，请关注社区Issue [#IBV35D](https://gitee.com/mindspore/mindformers/issues/IBV35D)。

<br/>

## Q: 如何生成模型切分策略文件？

A: 模型切分策略文件记录了模型权重在分布式场景下的切分策略，一般在离线权重切分时使用。在网络`yaml`文件中配置`only_save_strategy: True`，然后正常启动分布式任务，便可在`output/strategy/`目录下生成分布式策略文件，详细介绍请参阅[分布式权重切分与合并教程](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/ckpt.html)。

<br/>

## Q: 生成`ranktable`文件报错`socket.gaierror: [Errno -2] Name or service not known`或者`socket.gaierror: [Errno -3] Temporary failure in name resolution`，怎么解决？

A: 从`MindSpore Transformers r1.2.0`版本开始，集群启动统一使用`msrun`方式，`ranktable`启动方式已废弃。

<br/>

## Q: 通过源码安装MindSpore Transformers时依赖包下载速度慢，怎么解决？

A: `build.sh`使用[清华源](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)下载MindSpore Transformers所依赖的Python包。如需修改镜像源，可以修改`build.sh`中下载命令`pip install mindformers*whl -i https://pypi.tuna.tsinghua.edu.cn/simple` ，将`-i`后地址替换为目标镜像源地址。

<br/>