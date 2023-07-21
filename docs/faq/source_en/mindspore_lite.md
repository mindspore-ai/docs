# MindSpore Lite Use

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/faq/source_en/mindspore_lite.md)

<font size=3>**Q： What are the limitations of NPU?**</font>

A： Currently NPU only supports system ROM version EMUI>=11. Chip support includes Kirin 9000, Kirin 9000E, Kirin 990, Kirin 985, Kirin 820, Kirin 810, etc. For specific constraints and chip support, please see: <https://developer.huawei.com/consumer/en/doc/development/hiai-Guides/mapping-relationship-0000001052830507#EN-US_TOPIC_0000001052830507__section94427279718>

<font size=3>**Q： Why does the static library after cutting with the cropper tool fail to compile during integration?**</font>

A： Currently the cropper tool only supports CPU libraries, that is, `-e CPU` is specified in the compilation command. For details, please refer to [Use clipping tool to reduce library file size](https://www.mindspore.cn/tutorial/lite/en/r1.1/use/cropper_tool.html) document.
