# Sharding Techniques

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_en/parallel/split_technique.md)

## Overview

Given a new model, from the user's perspective, the key issue is to configure which operator's slicing strategy will yield better performance. Since the goal of strategy propagation is to minimize the cost of tensor rearranging rather than minimizing the end-to-end iteration time, it is important to configure the appropriate cut strategy for the "key operators". However, there is no explicit rule governing which operators must be configured with a sharding strategy. Nevertheless, based on our experience in training large models, there are some principles that can be used to guide new users in configuring parallel strategies. Here, we list 3 empirical principles.

### Configuring Operators Involving Weights

The sharding strategy for parameter weights is very important, especially for large models, as the memory consumption caused by parameter weights accounts for a large portion of the total memory consumption for model training. Therefore, operators involving weights usually need to explicitly configure the sharding strategy. In the two examples below, the Gather and MatMul operators involving weights are configured with sharding strategy, while the other operators are not. These correspond the data-parallel VocabEmbedding layer and hybrid-parallel FeedForward Layer in [mindformers](https://gitee.com/mindspore/mindformers/blob/master/mindformers/modules/transformer/transformer.py), respectively.

![sp_case1_zh](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/tutorials/experts/source_zh_cn/parallel/images/sp_case1_zh.png "Configuring Operators Involving Weights")

### Configuring Dimension-changing Operators

The operators of deep learning frameworks can be broadly categorized into two types: operators that are semantically simple and dimension-preserving and operators that change the dimension of the input tensor. For dimension-preserving operators, the strategy propagation algorithm can propagate the sharding strategy more easily. However, for dimension-changing operators, explicitly configuring the sharding strategy is the only way to better express the user initial thoughts and avoid the strategy propagation algorithm from deriving the sharding strategy that is not expected by the user. In the example below, ReduceMean and MatMul are dimension-changing operators that are configured with sharding strategy.

![sp_case2_zh](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/tutorials/experts/source_zh_cn/parallel/images/sp_case2_zh.png "Configuring Dimension-changing Operators")

### Configuring Boundary Operators that Change in Parallel Method

For ResNet-like models, different parts of the model have different preferred parallel: the first half uses data parallel, and the second half uses model parallel for optimal iterative performance. This can be accomplished by configuring strategy for boundary operators that change in a parallel method. In the example below, the first MatMul is configured with a strategy for data parallel, which will propagate the strategy for data parallel forward to the first half of the model, while the second MatMul is configured with a strategy for model parallel, which will propagate the strategy for model parallel backward to the second half of the model.

![sp_case3_zh](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/tutorials/experts/source_zh_cn/parallel/images/sp_case3_zh.png "Configuring Boundary Operators that Change in Parallel Method")

Users working with strategy propagation need to have some understanding not only of its propagation algorithm itself, but also of the parallelism of the model to be trained. If there exists a certain operator whose parallel strategy determined by the strategy propagation algorithm does not meet the user's expectations, that can always be solved by configuring an additional operator parallel strategy. In practice, for a new model, it does take several attempts to obtain an overall parallel configuration with better performance.

