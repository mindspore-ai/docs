# 端侧使用类

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/faq/source_zh_cn/mindspore_lite.md)

<font size=3>**Q：NPU推理存在什么限制？**</font>

A：目前NPU仅支持在系统ROM版本EMUI>=11、芯片支持包括Kirin 9000、Kirin 9000E、Kirin 990、Kirin 985、Kirin 820、Kirin 810等，具体约束和芯片支持请查看：<https://developer.huawei.com/consumer/cn/doc/development/hiai-Guides/mapping-relationship-0000001052830507#ZH-CN_TOPIC_0000001052830507__section94427279718>

<font size=3>**Q：为什么使用裁剪工具裁剪后的静态库在集成时存在编译失败情况？**</font>

A：目前裁剪工具仅支持CPU的库，即编译命令中指定了`-e CPU`，具体使用请查看[使用裁剪工具降低库文件大小](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.1/use/cropper_tool.html)文档。

