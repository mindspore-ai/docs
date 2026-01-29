# Release Notes

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1.post1/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/r2.7.1.post1/docs/vllm_mindspore/docs/source_zh_cn/release_notes/release_notes.md)

## vLLM-MindSpore插件 0.5.0 Release Notes

vLLM MindSpore插件0.5.0版本，配套vLLM 0.11.0版本。以下为此版本支持的关键新功能和模型。

### 新特性

- 架构适配：完成架构升级并适配 vLLM 0.11.0版本，全面继承升级前版本的所有功能；
- 性能优化：优化V1架构调度机制，提升大并发长输入场景下的推理性能；
- 图捕获功能支持：支持AclGraph图捕获功能。

### 新模型

- Qwen3-VL系列模型
- GLM系列模型
    - GLM-4.1V支持原生模型

### 问题修复
- 关于encoder的显存泄露问题
    - [对于vLLM的修复](https://github.com/vllm-project/vllm/pull/31857)
    - [对于vLLM-MindSpore插件的修复](https://atomgit.com/mindspore/vllm-mindspore/pull/1447)

### 贡献者

感谢以下人员做出的贡献：

alien_0119, candyhong, can-gaa-hou, ccsszz, cs123abc, dayschan, Erpim, fary86, hangangqiang, horcam, huandong, huzhikun, i-robot, jiahaochen666, JingweiHuang, lijiakun, liu lili, lvhaoyu, lvhaoyu1, moran, nashturing, one_east, panshaowu, pengjingyou, r1chardf1d0, tongl, TrHan, tronzhang, TronZhang, twc, uh, w00521005, wangpingan2, WanYidong, WeiCheng Tan, wusimin, yangminghai, yyyyrf, zhaizhiqiang, zhangxuetong, zhang_xu_hao1230, zhanzhan1, zichun_ye, zlq2020

欢迎以任何形式对项目提供贡献！
