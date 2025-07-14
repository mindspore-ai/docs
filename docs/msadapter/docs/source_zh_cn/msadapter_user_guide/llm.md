# 大模型开发与适配

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/msadapter/docs/source_zh_cn/msadapter_user_guide/llm.md)

大模型训练是一种非常复杂的过程，涉及到分布式并行领域许多技术和挑战，当前Megatron已经成为业界主流的大模型加速库。为满足用户大模型代码更快在MindSpore上迁移使用，MSAdapter当前版本已经支持MindSpeed加速库，兼容Megatron生态。当前已经支持DeepSeek/Qwen等主流模型，未来MSAdapter持续演进，支持更多业界主流生态模型。

希望学习使用MSAdapter进行大模型开发请参考：[MindSpeed MindSpore后端迁移开发指南](https://gitee.com/ascend/MindSpeed-Core-MS/blob/master/docs/develop_guide.md)。

此外，在MindSpeed加速库的基础上也提供了大语言模型、多模态模型套件加速库，用户可以安装MSAdapter及配套昇腾软件直接使用：

1. 大语言模型库：[MindSpeed-LLM](https://gitee.com/ascend/MindSpeed-LLM/blob/master/docs/mindspore)

2. 多模态模型库：[MindSpeed-MM](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/mindspore/getting_start.md)