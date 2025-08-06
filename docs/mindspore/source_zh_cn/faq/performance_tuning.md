# 性能调优

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/faq/performance_tuning.md)

## Q: MindSpore安装完成，执行训练时发现网络性能异常，权重初始化耗时过长，怎么办？  

A：可能与环境中使用了`scipy 1.4`系列版本有关，通过`pip list | grep scipy`命令可查看scipy版本，建议改成MindSpore要求的`scipy`版本。版本第三方库依赖可以在`requirement.txt`中查看。
<https://gitee.com/mindspore/mindspore/blob/br_base/requirements.txt>

<br/>

## Q: 在昇腾芯片上进行模型训练时，如何选择batchsize达到最佳性能效果？  

A：在昇腾芯片上进行模型训练时，在batchsize等于AI CORE个数或倍数的情况下可以获取更好的训练性能。AI CORE个数可通过链接中的命令行进行查询。
<https://support.huawei.com/enterprise/zh/doc/EDOC1100206828/eedfacda>
