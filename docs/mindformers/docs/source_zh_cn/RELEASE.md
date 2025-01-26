# Release Notes

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.10/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.10/docs/mindformers/docs/source_zh_cn/RELEASE.md)

## MindSpore Transformers 1.3.2 Release Notes

以下为 MindSpore Transformers (以下称为 MindFormers) 套件 1.3.2 版本的变更日志，相较于1.3.0版本有以下关键新特性和 bugfix 。

### 新特性

- [Loss](https://gitee.com/mindspore/mindformers/pulls/4709)：更新流水线并行条件下的loss计算逻辑，使得能够与Megatron对齐。
- [可靠性](https://gitee.com/mindspore/mindformers/pulls/4629)：支持通过local-loss快速识别故障卡的功能。
- [高性能](https://gitee.com/mindspore/mindformers/pulls/4630)：支持使用特定rank进行profiling。
- [日志](https://gitee.com/mindspore/mindformers/pulls/4622)：增加在训练场景下打印保存ckpt时的时长信息的功能。
- [稳定性](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.0/function/resume_training.html#%E6%96%AD%E7%82%B9%E7%BB%AD%E8%AE%AD)：断点续训支持通信阻塞，完成权重文件一致性校验。

### 新模型

以下为新支持模型：

| 模型                                                                                             | 规格                                                                |
|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [TeleChat2](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/telechat2) | TeleChat2-7b (微调、推理)、TeleChat2-35b (微调、推理)、TeleChat2-115b (微调、推理) |

### Bugfix

在当前版本发布周期内，我们进行了模型/功能/易用性/文档等诸多方面的 bugfix ，在此仅列举部分修复内容：

- [!5051](https://gitee.com/mindspore/mindformers/pulls/5051)：修复训练或推理时，lora的dtype与model_config中不一致的问题。
- [!5021](https://gitee.com/mindspore/mindformers/pulls/5021)：修复benchmark训练工具在线下载处理数据集报错的问题。
- [!4999](https://gitee.com/mindspore/mindformers/pulls/4999)：修复多batch并发场景下SLoraModel无法改变输入batchsize的问题。
- [!4914](https://gitee.com/mindspore/mindformers/pulls/4914)：修复部分场景使用msrun启动MF_LOG_PATH失效的问题。

### 贡献者

感谢以下人员做出的贡献：

Chong Li、chenyijie、heqinglin、huangshengshuai、lilei、lizhihao、lizheng、moran、paolo poggi、wangshaocong、wutiancheng、xiaoshihan、yangminghai、yangzhenzhang、zhanzhan、zhaozhengquan、ZhouJingfeng、zhouyaqiang、包淦超、常少中、陈心锐、陈昱坤、陈志坚、程鹏、楚浩田、戴仁杰、冯浩、冯明昊、冯汛、耿辰华、郭儒辰、古雅诗、贺冬冬、何泽泉、胡思超、胡映彤、宦晓玲、黄磊、黄新元、黄勇、黄子灵、纪泽伟、金仁操、孔德硕、孔紫怡、寇凯睿、蓝翔、李俊标、李洋、李文、李永文、李子垠、林鑫、林盈来、刘晨晖、刘奇、刘烙彬、刘力力、刘思铭、吕凯盟、倪钰鑫、牛君豪、邱杨、任峪瑾、赛尧、孙宇轩、唐德志、谭纬城、王浩然、汪家傲、王嘉霖、王廖辉、王双玲、魏琢艺、吴治锋、吴致远、吴昊天、杨星宇、杨犇、杨承翰、杨璇、易阳、尤日帆、俞涵、张浩、张泓铨、张吉昊、张俊杰、张敏利、张森镇、张伟、张一飞、张奕晖、张雨强、赵奕舜、周洪叶、周声煦、周小琪、朱亿超、邹文祥

欢迎以任何形式对项目提供贡献！
