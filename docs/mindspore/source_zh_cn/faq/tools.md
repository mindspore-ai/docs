# 工具

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/faq/tools.md)

## 常见问题

### Q: 使用溢出检测Dump功能时，遇到`RuntimeError: aclnnAllFiniteGetWorkspaceSize call failed, please check!`报错，该如何解决？

A: 该错误通常是因为溢出检测功能所依赖的自定义算子与当前 CANN 版本不兼容所致。MindSpore 的溢出检测 Dump 功能对 CANN 版本有严格要求，高版本 MindSpore 无法兼容低版本 CANN。

建议通过以下方式解决：

- 升级CANN版本

  请根据您使用的 MindSpore 版本，参考官方文档中的 [版本配套关系表](https://www.mindspore.cn/install)，安装匹配或更高版本的 CANN。
- 改用统计量Dump或其他Dump方式

  若暂时无法升级 CANN ，可关闭溢出检测Dump，转而使用：
  - 统计量 Dump：记录张量的最大值、最小值等信息，用于判断是否发生溢出；
  - 全量 Dump 或选择性 Dump：保存中间张量数据，辅助离线分析数值异常，详细请参考[Dump功能调试](https://www.mindspore.cn/tutorials/zh-CN/master/debug/dump.html)。

通过上述方法，可有效规避该运行时错误，并继续完成溢出问题的排查。
