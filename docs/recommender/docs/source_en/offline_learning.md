# Offline Training

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/recommender/docs/source_en/offline_learning.md)

## Overview

One of the main challenges of recommendation model training is the storage and training of large-scale feature vectors. MindSpore Recommender provides a perfect solution for training large-scale feature vectors for offline scenarios.

## Overall Architecture

The training architecture for large-scale feature vectors in recommendation models is shown in the figure below, in which the core adopts the technical scheme of distributed multi-level Embedding Cache. The distributed parallel technology of multi-machine and multi-card based on model parallelism implements large-scale and low-cost  recommendation training of large models.

![image.png](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/docs/recommender/docs/source_en/images/offline_training.png)

## Example

[Wide&Deep distributed training](https://github.com/mindspore-lab/mindrec/tree/r0.3/models/wide_deep)
