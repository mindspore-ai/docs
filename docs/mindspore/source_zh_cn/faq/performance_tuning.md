# 性能调优

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/faq/performance_tuning.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

<font size=3>**Q: MindSpore安装完成，执行训练时发现网络性能异常，权重初始化耗时过长，怎么办？**</font>  

A: 可能与环境中使用了`scipy 1.4`系列版本有关，通过`pip list | grep scipy`命令可查看scipy版本，建议改成MindSpore要求的`scipy`版本。版本第三方库依赖可以在`requirement.txt`中查看。
<https://gitee.com/mindspore/mindspore/blob/{version}/requirements.txt>
> 其中version替换为MindSpore具体的版本分支。
