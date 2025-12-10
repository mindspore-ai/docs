# Release Notes

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.2/docs/vllm_mindspore/docs/source_zh_cn/release_notes/release_notes.md)

## vLLM-MindSpore插件 0.4.0 Release Notes

vLLM MindSpore插件0.4.0版本，配套vLLM 0.9.1版本。以下为此版本支持的关键新功能和模型。

### 新特性

- 架构适配：完成架构升级并适配 vLLM 0.9.1版本，全面继承升级前版本的所有功能；新增支持通过 Ray 方式部署 DP 并行服务，具体配置及操作说明请参考[多机并行推理](../getting_started/tutorials/deepseek_parallel/deepseek_r1_671b_w8a8_dp4_tp4_ep4.md)；
- 量化支持：DeepSeek-R1 模型已支持 A8W4 量化推理功能，相关技术细节与使用指南详见 [DeepSeek-R1 A8W4量化推理模型链接](https://modelers.cn/models/MindSpore-Lab/R1-A8W4)；
- 性能优化：集成FA3量化推理、MLA系列算子，显著提升DeepSeek系列模型长序列场景下的运行性能，部分场景性能提升幅度超 10%；
- 易用性提升：优化 MindSpore Transformers 后端模型服务化部署流程，不再依赖 YAML 配置文件，用户可直接基于社区开源的 Hugging Face 模型配置文件完成部署操作。

### 新模型

- DeepSeek系列模型
    - DeepSeek-R1 A8W4 量化推理
- TeleChat系列模型
    - TeleChat2模型推理
- GLM系列模型
    - GLM-4模型推理
    - GLM-4.5模型推理
- Qwen3系列模型
    - Qwen3原生模型
- Qwen2,5系列模型
    - Qwen2.5-VL支持原生模型

### 贡献者

感谢以下人员做出的贡献：

alien_0119、candyhong、can-gaa-hou、ccsszz、cs123abc、dayschan、Erpim、fary86、hangangqiang、horcham_zhq、huandong、huzhikun、i-robot、jiahaochen666、JingweiHuang、lijiakun、liu lili、lvhaoyu、lvhaoyu1、moran、nashturing、one_east、panshaowu、pengjingyou、r1chardf1d0、tongl、TrHan、tronzhang、TronZhang、twc、uh、w00521005、wangpingan2、WanYidong、WeiCheng Tan、wusimin、yangminghai、yyyyrf、zhaizhiqiang、zhangxuetong、zhang_xu_hao1230、zhanzhan1、zichun_ye、zlq2020

欢迎以任何形式对项目提供贡献！
