# 情感分类

## 概述

在隐私合规场景下，通过端云协同的联邦学习建模方式，可以充分发挥端侧数据的优势，避免用户敏感数据直接上报云侧。在联邦学习应用场景的探索中，输入法场景引起了我们的注意。由于用户在使用输入法时对自己的文字隐私十分看重，并且输入法上的智慧功能也是用户非常需要的。因此，联邦学习天然适用在输入法场景中。MindFL将联邦语言模型应用到了输入法的表情图片预测功能中。联邦语言模型会根据聊天文本数据推荐出适合当前语境的表情图片。在使用联邦学习建模时，每一张表情图片会被定义为一个情感标签类别，而每个聊天短语会对应一个表情图片。MindFL将表情图片预测任务定义为联邦情感分类任务。

## 准备环节

### 环境

参考：[服务端环境配置](https://gitee.com/mindspore/docs/tree/master/docs/mind_fl/source_zh_cn/deploy_mind_fl_cluster.md)和[客户端环境配置](https://gitee.com/mindspore/docs/tree/master/docs/mind_fl/source_zh_cn/deploy_FL_Client.md)。

### 数据

[用于训练的数据](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/train.tar.gz)包含100个用户聊天文件，其目录结构如下：

```text
mobile/datasets/train/
├── 0.tsv  # 用户0的训练数据
├── 1.tsv  # 用户1的训练数据
│
│          ......
│
└── 99.tsv  # 用户99的训练数据
```

[用于验证的数据](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/eval.tar.gz)包含1个聊天文件，其目录结构如下：

```text
mobile/datasets/eval/
├── 0.tsv  # 验证数据
```

[标签对应的表情图片数据](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/memo.tar.gz)包含107个图片，其目录结构如下：

```text
mobile/datasets/memo/
├── 0.gif  # 第0个标签对应的表情图片
├── 1.gif  # 第1个标签对应的表情图片
│
│          ......
│
└── 106.gif  # 第106个标签对应的表情图片
```

### 模型相关文件

生成模型需要的起始的[checkpoint](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/models/albert_init.ckpt)和[词典](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/models/vocab.txt)的目录结构如下：

```text
mobile/models/
├── albert_init.ckpt  # 起始的checkpoint
└── vocab.txt  # 词典
```

## 定义网络

联邦学习中的语言模型使用ALBERT模型[1]。客户端上的ALBERT模型包括：embedding层、encoder层和classifier层。

具体网络定义请参考[源码](https://gitee.com/mindspore/mindspore/tree/master/tests/st/fl/mobile/src/model.py)。

### 生成端侧模型文件

#### 将模型导出为MindIR格式文件

代码如下：

```python
import numpy as np
from mindspore import export, Tensor
from src.config import train_cfg, client_net_cfg
from src.cell_wrapper import NetworkTrainCell

# 构建模型
client_network_train_cell = NetworkTrainCell(client_net_cfg)

# 构建输入数据
input_ids = Tensor(np.zeros((train_cfg.batch_size, client_net_cfg.seq_length), dtype=np.int32))
attention_mask = Tensor(np.zeros((train_cfg.batch_size, client_net_cfg.seq_length), dtype=np.int32))
token_type_ids = Tensor(np.zeros((train_cfg.batch_size, client_net_cfg.seq_length), dtype=np.int32))
label_ids = Tensor(np.zeros((train_cfg.batch_size, client_net_cfg.num_labels), dtype=np.int32))

# 导出模型
export(client_network_train_cell, input_ids, attention_mask, token_type_ids, label_ids, file_name='albert_train.mindir', file_format='MINDIR')
```

#### 将MindIR文件转化为联邦学习端侧框架可用的ms文件

参考[实现一个图像分类应用](https://gitee.com/mindspore/docs/tree/master/docs/mind_fl/source_zh_cn/image_classification_application.md)中生成端侧模型文件部分。

## 启动联邦学习流程

### 模型训练

首先在服务端启动脚本：参考[Federated Learning Server集群部署方式](https://gitee.com/mindspore/docs/tree/master/docs/mind_fl/source_zh_cn/deploy_mind_fl_cluster.md)

然后在客户端启动脚本：参考[Android启动联邦学习步骤](https://gitee.com/mindspore/docs/tree/master/docs/mind_fl/source_zh_cn/fl_android_application.md)

### 模型验证

在客户端启动脚本：参考[Android启动联邦学习步骤](https://gitee.com/mindspore/docs/tree/master/docs/mind_fl/source_zh_cn/fl_android_application.md)

## 实验结果

联邦学习总迭代数为5，客户端本地训练epoch数为10，batchSize设置为16。

|        | Top1精度 | Top5精度 |
| ------ | -------- | -------- |
| ALBERT | 24%      | 70%      |

## 参考文献

[1] Lan Z ,  Chen M ,  Goodman S , et al. ALBERT: A Lite BERT for Self-supervised Learning of Language Representations[J].  2019.

