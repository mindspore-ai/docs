# Release Notes

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/feature/atomgit/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/release_notes/release_notes.md)

## vLLM-MindSpore Plugin 0.4.0 Release Notes

The vLLM-MindSpore Plugin 0.4.0 version is compatible with vLLM 0.9.1. Below are the new features and models supported in this release.

### New Features

- Architecture Adaptation: Completed architectural upgrades and adapted to vLLM 0.9.1, fully inheriting all features of the older version. Added support for deploying DP parallel services by Ray. For specific configurations and operational instructions, please refer to the [Multi-Machine Parallel Inference](../getting_started/tutorials/deepseek_parallel/deepseek_r1_671b_w8a8_dp4_tp4_ep4.md).  
- Quantization Support: The DeepSeek-R1 model now supports A8W4 quantization inference. For technical details and usage guidelines, see [DeepSeek-R1 A8W4 Quantization Inference Model Link](https://modelers.cn/models/MindSpore-Lab/R1-A8W4).  
- Performance Optimization: Integrated FA3 quantization inference and MLA series operators, significantly improving the performance of the DeepSeek model series in long-sequence scenarios. Performance improvements exceed 10% in some scenarios.  
- Usability Enhancements: Optimized the deployment process for backend model services in MindSpore Transformers, eliminating the dependency on YAML configuration files. Users can now complete deployments directly using community-open-sourced Hugging Face model configuration files.  

### New Models

- DeepSeek Series Model:
    - DeepSeek-R1 A8W4 Quantization Inference
- TeleChat Model Series:
    - TeleChat2 Model Inference
- GLM Model Series:
    - GLM-4 Model Inference
    - GLM-4.5 Model Inference
- Qwen3 Model Series:
    - Qwen3 Native Model
- Qwen2.5 Model Series:
    - Qwen2.5-VL Native Model

### Contributors

Thanks to the following contributors for their efforts:

alien_0119, candyhong, can-gaa-hou, ccsszz, cs123abc, dayschan, Erpim, fary86, hangangqiang, horcam, huandong, huzhikun, i-robot, jiahaochen666, JingweiHuang, lijiakun, liu lili, lvhaoyu, lvhaoyu1, moran, nashturing, one_east, panshaowu, pengjingyou, r1chardf1d0, tongl, TrHan, tronzhang, TronZhang, twc, uh, w00521005, wangpingan2, WanYidong, WeiCheng Tan, wusimin, yangminghai, yyyyrf, zhaizhiqiang, zhangxuetong, zhang_xu_hao1230, zhanzhan1, zichun_ye, zlq2020

Contributions to the project in any form are welcome!
