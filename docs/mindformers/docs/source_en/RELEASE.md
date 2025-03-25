# Release Notes

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.10/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.10/docs/mindformers/docs/source_en/RELEASE.md)

## MindSpore Transformers 1.3.2 Release Notes

Below is the changelog for MindSpore Transformers (referred to as MindFormers) version 1.3.2, highlighting key new features and bug fixes compared to version 1.3.0.

### New Features

- [Loss](https://gitee.com/mindspore/mindformers/pulls/4709)：Update the loss calculation logic under pipeline parallelism to be consistent with Megatron.
- [Reliability](https://gitee.com/mindspore/mindformers/pulls/4629)：Support quick identification of faulty cards through local-loss.
- [High Performance](https://gitee.com/mindspore/mindformers/pulls/4630)：Support for profiling using a specific rank.
- [Log](https://gitee.com/mindspore/mindformers/pulls/4622)：Add the function of printing the duration information when saving ckpt in training scenarios.
- [Stability](https://www.mindspore.cn/mindformers/docs/en/r1.3.0/function/resume_training.html#resumable-training)：Resume training supports communication blocking and completes weight file consistency verification.

### New Models

The following new models are now supported:

| Model                                                                                   | Specifications                                                                                                |
|-------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| [TeleChat2](https://gitee.com/mindspore/mindformers/tree/v1.3.2/research/telechat2) | TeleChat2-7b (finetune, inference), TeleChat2-35b (finetune, inference), TeleChat2-115b (finetune, inference) |

### Bugfix

During this release cycle, we addressed numerous bugs across models, functionalities, usability, and documentation.
Here are some notable fixes:

- [!5051](https://gitee.com/mindspore/mindformers/pulls/5051)：Fixed the issue where the dtype of lora did not match the one in model_config when training or inference.
- [!5021](https://gitee.com/mindspore/mindformers/pulls/5021)：Fixed the issue that benchmark training tool online download processing dataset reported an error.
- [!4999](https://gitee.com/mindspore/mindformers/pulls/4999)：Fix the problem that SLoraModel cannot change the input batchsize in multi batch concurrent scenarios.
- [!4914](https://gitee.com/mindspore/mindformers/pulls/4914)：Fixed the issue where MF_LOG_PATH failed to start using msrun in some scenarios.

### Contributors

Thanks to the following individuals for their contributions:

Chong Li, chenyijie, heqinglin, huangshengshuai, lilei, lizhihao, lizheng, moran, paolo poggi, wangshaocong, wutiancheng, xiaoshihan, yangminghai, yangzhenzhang, zhanzhan, zhaozhengquan, ZhouJingfeng, zhouyaqiang, 包淦超, 常少中, 陈心锐, 陈昱坤, 陈志坚, 程鹏, 楚浩田, 戴仁杰, 冯浩, 冯明昊, 冯汛, 耿辰华, 郭儒辰, 古雅诗, 贺冬冬, 何泽泉, 胡思超, 胡映彤, 宦晓玲, 黄磊, 黄新元, 黄勇, 黄子灵, 纪泽伟, 金仁操, 孔德硕, 孔紫怡, 寇凯睿, 蓝翔, 李俊标, 李洋, 李文, 李永文, 李子垠, 林鑫, 林盈来, 刘晨晖, 刘奇, 刘烙彬, 刘力力, 刘思铭, 吕凯盟, 倪钰鑫, 牛君豪, 邱杨, 任峪瑾, 赛尧, 孙宇轩, 唐德志, 谭纬城, 王浩然, 汪家傲, 王嘉霖, 王廖辉, 王双玲, 魏琢艺, 吴治锋, 吴致远, 吴昊天, 杨星宇, 杨犇, 杨承翰, 杨璇, 易阳, 尤日帆, 俞涵, 张浩, 张泓铨, 张吉昊, 张俊杰, 张敏利, 张森镇, 张伟, 张一飞, 张奕晖, 张雨强, 赵奕舜, 周洪叶, 周声煦, 周小琪, 朱亿超, 邹文祥

Contributions to the project in any form are welcome!
