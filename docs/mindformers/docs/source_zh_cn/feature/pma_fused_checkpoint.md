# Pre-trained Model Average 权重合并

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/docs/mindformers/docs/source_zh_cn/feature/pma_fused_checkpoint.md)

## 概述

Pre-trained Model Average（PMA）权重合并是指在训练过程中，根据选择 Exponential Moving Average（EMA）算法或 Simple Moving Average（SMA）算法对权重进行融合合并，从而提升模型训练的效果。

MindSpore Transformers提供了`EMA`算法和`SMA`算法对权重进行融合合并，合并公式如下：

EMA算法公式：$PMA_n = (1 - \alpha) \times PMA_{n-1} + \alpha \times W_n$

> EMA算法通过指数递减的方式分配权重，对最近的模型权重更为敏感，能够快速响应模型在训练后期的变化。

SMA算法公式：$PMA_n = (W_1 + ... + W_n) / n$

> SMA算法在所有模型权重上均匀分配权重，对待每个权重都一视同仁。

| 参数名称        | 参数说明                 |
|-------------|----------------------|
| $PMA_n$     | 第n步的合并权重             |
| $PMA_{n-1}$ | 第n-1步的合并权重           |
| $W_1$       | 第1步的原始权重             |
| $W_n$       | 第n步的原始权重             |
| $\alpha$    | 融合系数，只有当算法选择EMA时才会生效 |
| $n$         | 表示n个权重取平均值           |

> - 模型在训练时，会每隔固定步数选取一个权重进行公式计算，并作为中间值`pma_weight`保存在权重中，此时并不会影响原来权重的参数取值。
> - 当选取的权重数量达到设定的数量时，权重中间值`pma_weight`写入并覆盖原参数取值后置零，训练进入下一个周期的权重合并。

参考文献如下：

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

## 使用方法

**注意**：以下示例所展示的参数数值仅作为实验数据，请以真实训练数据为准。

本功能通过YAML配置文件使能：

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

**参数说明：**

| 参数名称            | 描述                                                                  | 类型                              | 是否可选       | 取值范围           |
|-----------------|---------------------------------------------------------------------|---------------------------------|------------|----------------|
| type            | 优化器类型，启用PMA特性需要设定为`PmaAdamW`。默认值为`AdamW`。                           | String                          | 可选         |                |
| betas           | `moment1`、 `moment2` 的指数衰减率。每个参数范围为（0.0,1.0）。默认值为`(0.9, 0.999)` 。 | Union[list(float), tuple(float)] |   可选         | (0.0,1.0)      |
| eps             | 将添加到分母中，以提高数值稳定性。必须大于0。默认值为 `1e-6` 。                              | float                           |     可选       | 正数             |
| weight_decay    | 设定优化器权重衰减系数。默认值为`0.0`。                                              | float                           |     可选       |                |
| fused_num       | 设定`fused_num`个权重进行融合，根据融合算法将融合后的权重更新到网络参数中。默认值为`10`。                | int                             | 可选         | 正整数            |
| interleave_step | 选取待融合权重的step间隔数，每`interleave_step`个step取一次权重作为候选权重进行融合。默认值为`1000`。  | int                             | 可选         | 正整数            |
| fused_algo      | 融合算法，支持`ema`和`sma`。默认值为`ema`。                                       | string                          | 可选         | [`ema`, `sma`] |
| ema_alpha       | 融合系数，仅在`fused_algo`=`ema`时生效。默认值为`0.2`。                             | float                           | 可选    | (0, 1)         |

### PmaAdamW优化器配置介绍

有关PmaAdamW优化器配置相关内容，可参见 [MindSpore Transformers PmaAdamW 源码](https://gitee.com/mindspore/mindformers/blob/r1.7.0/mindformers/core/optim/pma_adamw.py) 的相关链接。
