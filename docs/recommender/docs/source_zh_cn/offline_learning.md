# 离线训练

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/recommender/docs/source_zh_cn/offline_learning.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

推荐模型训练的主要挑战之一是对于大规模特征向量的存储与训练，MindSpore Recommender为离线场景的大规模特征向量训练提供了完善的解决方案。

## 整体架构

针对推荐模型中大规模特征向量的训练架构如下图所示，其中核心采用了分布式多级Embedding Cache的技术方案，同时基于模型并行的多机多卡分布式并行技术，实现了大规模低成本的推荐大模型训练。

![image.png](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/recommender/docs/source_zh_cn/images/offline_training.png)

## 使用样例

[Wide&Deep 分布式训练](https://gitee.com/mindspore/recommender/tree/r2.0.0-alpha/models/wide_deep)