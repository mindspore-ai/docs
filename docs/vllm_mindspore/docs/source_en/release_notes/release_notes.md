# Release Notes

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1.post1/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/r2.7.1.post1/docs/vllm_mindspore/docs/source_en/release_notes/release_notes.md)

## vLLM-MindSpore Plugin 0.5.0 Release Notes

The vLLM-MindSpore Plugin 0.5.0 version is compatible with vLLM 0.11.0. Below are the new features and models supported in this release.

### New Features

- Architecture Adaptation: Completed architectural upgrades and adapted to vLLM 0.11.0, fully inheriting all features of the older version.  
- Performance Optimization: Enhanced the V1 architecture scheduling to improve inference performance in high-concurrency, long-input scenarios.
- Graph Capture Feature Support: AclGraph graph capture functionality is supported.

### New Models

- Qwen3-VL Model Series
- GLM Model Series:
    - GLM-4.1V Native Model

### Problem Fixes

- Encoder Memory Leak Issue
    - [Fix for vLLM](https://github.com/vllm-project/vllm/pull/31857)
    - [Fix for vLLM-MindSpore Plugin](https://atomgit.com/mindspore/vllm-mindspore/pull/1447)


### Contributors

Thanks to the following contributors for their efforts:

alien_0119, candyhong, can-gaa-hou, ccsszz, cs123abc, dayschan, Erpim, fary86, hangangqiang, horcam, huandong, huzhikun, i-robot, jiahaochen666, JingweiHuang, lijiakun, liu lili, lvhaoyu, lvhaoyu1, moran, nashturing, one_east, panshaowu, pengjingyou, r1chardf1d0, tongl, TrHan, tronzhang, TronZhang, twc, uh, w00521005, wangpingan2, WanYidong, WeiCheng Tan, wusimin, yangminghai, yyyyrf, zhaizhiqiang, zhangxuetong, zhang_xu_hao1230, zhanzhan1, zichun_ye, zlq2020

Contributions to the project in any form are welcome!
