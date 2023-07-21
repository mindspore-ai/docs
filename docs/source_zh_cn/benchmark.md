# 基准性能

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.5/docs/source_zh_cn/benchmark.md)

本文介绍MindSpore的基准性能。MindSpore预训练模型可参考[Model Zoo](https://gitee.com/mindspore/mindspore/tree/r0.5/model_zoo)。

## 训练性能

### ResNet

| Network | Network Type | Dataset | MindSpore Version | Resource &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Precision | Batch Size | Throughput | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 v1.5 | CNN | ImageNet2012 | 0.5.0-beta | Ascend: 1 * Ascend 910 </br> CPU：24 Cores | Mixed | 256 | 2115 images/sec | - |
|  |  |  |  | Ascend: 8 * Ascend 910 </br> CPU：192 Cores | Mixed | 256 | 16600 images/sec | 0.98 |
|  |  |  |  | Ascend: 16 * Ascend 910 </br> CPU：384 Cores | Mixed | 256 | 32768 images/sec | 0.96 |

1. 以上数据基于华为云AI开发平台ModelArts测试获得，是训练过程整体下沉至Ascend 910 AI处理器执行所得的平均性能。
2. 业界其他开源框架数据可参考：[ResNet-50 v1.5 for TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5)。

### BERT

| Network |	Network Type | Dataset | MindSpore Version | Resource &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Precision | Batch Size | Throughput |  Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BERT-Large | Attention | zhwiki | 0.5.0-beta | Ascend: 1 * Ascend 910 </br> CPU：24 Cores | Mixed | 96 | 269 sentences/sec | - |
|  |  |  |  | Ascend: 8 * Ascend 910 </br> CPU：192 Cores | Mixed | 96 | 2069 sentences/sec | 0.96 |

1. 以上数据基于华为云AI开发平台ModelArts测试获得，其中网络包含24个隐藏层，句长为128个token，字典表包含21128个token。  
2. 业界其他开源框架数据可参考：[BERT For TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)。