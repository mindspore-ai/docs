# 使用MindConverter迁移模型定义脚本

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindinsight/docs/source_zh_cn/migrate_3rd_scripts_mindconverter.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 工具概述

MindConverter是一款模型迁移工具，可将PyTorch(ONNX)或Tensorflow(PB)模型快速迁移到MindSpore框架下使用。模型文件（ONNX/PB）包含网络模型结构（`network`）与权重信息（`weights`），迁移后将生成MindSpore框架下的模型定义脚本（`model.py`）与权重文件（`ckpt`）。

![mindconverter-overview](images/mindconverter-overview.png)

此外，本工具支持通过在PyTorch网络脚本中增加API(`pytorch2mindspore`)的方式，将PyTorch网络模型迁移到MindSpore框架下。

> - 由于战略调整，MindConverter从1.9.0开始将不再演进，官网文档及代码将逐步下架，请知悉。
> - 如对MindConverter项目感兴趣，请移步1.7.0版本（文档详见[MindConverter 1.7.0](https://www.mindspore.cn/mindinsight/docs/zh-CN/r1.7/migrate_3rd_scripts_mindconverter.html)）。
> - MindConverter当前仅维护1.6.0和1.7.0两个版本，后续维护工作也将逐步向1.7.0倾斜。
