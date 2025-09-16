# Pre-trained Model Average Weight Consolidation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/pma_fused_checkpoint.md)

## Overview

Pre-trained Model Average (PMA) weight merging refers to the process of merging weights based on the selection of Exponential Moving Average (EMA) or Simple Moving Average (SMA) algorithm during training, in order to enhance the effectiveness of model training.

MindSpore Transformers provides the `EMA` and `SMA` algorithms for weight fusion and merging. The merging formula is as follows:

EMA algorithm formula: $PMA_n = (1 - \alpha) \times PMA_{n-1} + \alpha \times W_n$

> The EMA algorithm allocates weights in an exponentially decreasing manner, making it more sensitive to the weights of the nearest model and able to quickly respond to changes in the model during the later stages of training.

SMA algorithm formula: $PMA_n = (W_1+ ... + Wn) / n$

> The SMA algorithm evenly distributes weights across all model weights and treats each weight equally.

| Parameter   | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| $PMA_n$     | The fused weight in step n                                                  |
| $PMA_{n-1}$ | The fused weight of step n-1                                                |
| $W_1$       | The original weight of step 1                                               |
| $W_n$       | The original weight of step n                                               |
| $\alpha$    | The fusion coefficient will only take effect when the algorithm chooses EMA |
| $n$         | Take the average of n weights                                               |

> The model will select a weight every fixed number of steps for formula calculation during training and save it as the middle value `pma_weight` in the weight, which will not affect the parameter values of the original weight.
> When the number of selected weights reaches the set number, the middle value of the weights `pma_weight` is written and overwritten with the zero after the original parameter value, and the training enters the next cycle of weight merging.

The reference is as follows:

```text
@misc{modelmerging,
      title={Model Merging in Pre-training of Large Language Models},
      authors={Yunshui Li, Yiyuan Ma, Shen Yan, Chaoyi Zhang, Jing Liu, Jianqiao Lu,
      Ziwen Xu, Mengzhao Chen, Minrui Wang, Shiyi Zhan, Jin Ma, Xunhao Lai, Deyi Liu, Yao Luo,
      Xingyan Bin, Hongbin Ren, Mingji Han, Wenhao Hao, Bairen Yi, LingJun Liu, Bole Ma,
      Xiaoying Jia, Xun Zhou, Siyuan Qiao, Liang Xiang, Yonghui Wu},
      year={2025},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.12082}
}
```

## Usage

**Note**: The parameter values shown in the following examples are only experimental data, please refer to real training data.

This feature is enabled through YAML configuration files:

```yaml
optimizer:
  type: PmaAdamW
  betas: [0.9, 0.999]
  eps: 1.e-6
  weight_decay: 0.0
  fused_num: 10
  interleave_step: 1000
  fused_algo: 'ema'
  ema_alpha: 0.2
```

**Parameter:**

| Parameter            | Description                                                                                                                                                              | Type                            | Optional       | Value Range           |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|------------|----------------|
| type            | Optimizer type, to enable PMA feature, it needs to be set to `PmaAdamW`. Default to `AdamW`.                                                                             | String                          | Optional         |                |
| betas           | The exponential decay rate of `moment1` and `moment2`. Each parameter range (0.0, 1.0). Default to ``(0.9, 0.999)``.                                                     | Union[list(float), tuple(float)] |   Optional         | (0.0,1.0)      |
| eps             | Add it to the denominator to improve numerical stability. Must be greater than 0. Default to ``1e-6``.                                                                   | float                           |     Optional       | positive number             |
| weight_decay    | Set the optimizer weight decay coefficient. Default to `0.0`.                                                                                                            | float                           |     Optional       |                |
| fused_num       | Set `fused_num` weights for fusion, and update the fused weights to the network parameters according to the fusion algorithm. Default to `10`.                          | int                             | Optional         | Positive integer            |
| interleave_step | Select the number of step intervals for the weights to be fused, and take a weight as a candidate weight for fusion once every `interleave_step` step. Default to `1000`. | int                             | Optional         | Positive integer            |
| fused_algo      | Fusion algorithm, supports `ema` and `sma`. Default to `ema`.                                                                                                            | string                          | Optional         | [`ema`, `sma`] |
| ema_alpha       | The fusion coefficient is only effective when `fused_algo` is set to `ema`. Default to `0.2`.                                                                            | float                           | Optional    | (0, 1)         |

### PmaAdamW Optimizer Configuration Introduction

For information on configuring the PmaAdamW optimizer, please refer to [MindSpore Transformers PmaAdamW Source Code](https://gitee.com/mindspore/mindformers/blob/master/mindformers/core/optim/pma_adamw.py).
