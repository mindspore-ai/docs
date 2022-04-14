# 基准性能

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/benchmark.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

本文介绍MindSpore的基准性能。MindSpore网络定义可参考[ModelZoo](https://gitee.com/mindspore/models/tree/master)。

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

| Network | Network Type | Dataset | MindSpore Version | Resource &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Precision | Batch Size | Throughput |  Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BERT-Large | Attention | zhwiki | 0.5.0-beta | Ascend: 1 * Ascend 910 </br> CPU：24 Cores | Mixed | 96 | 269 sentences/sec | - |
|  |  |  |  | Ascend: 8 * Ascend 910 </br> CPU：192 Cores | Mixed | 96 | 2069 sentences/sec | 0.96 |

1. 以上数据基于华为云AI开发平台ModelArts测试获得，其中网络包含24个隐藏层，句长为128个token，字典表包含21128个token。  
2. 业界其他开源框架数据可参考：[BERT For TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)。

### Wide & Deep (数据并行)

| Network | Network Type | Dataset | MindSpore Version | Resource &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Precision | Batch Size | Throughput |  Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wide & Deep | Recommend | Criteo | 0.6.0-beta | Ascend: 1 * Ascend 910 </br> CPU：24 Cores | Mixed | 16000 | 796892 samples/sec | - |
|  |  |  |  | Ascend: 8 \* Ascend 910 </br> CPU：192 Cores | Mixed | 16000*8 | 4872849 samples/sec | 0.76 |

1. 以上数据基于Atlas 800测试获得，且网络模型为数据并行。
2. 业界其他开源框架数据可参考：[Wide & Deep For TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep)。

### Wide & Deep (Host-Device混合计算模型并行)

| Network | Network Type | Dataset | MindSpore Version | Resource &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Precision | Batch Size | Throughput |  Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wide & Deep | Recommend | Criteo | 0.6.0-beta | Ascend: 1 * Ascend 910 </br> CPU：24 Cores | Mixed | 8000 | 68715 samples/sec | - |
|  |  |  |  | Ascend: 8 \* Ascend 910 </br> CPU：192 Cores | Mixed | 8000*8 | 283830 samples/sec | 0.51 |
|  |  |  |  | Ascend: 16 \* Ascend 910 </br> CPU：384 Cores | Mixed | 8000*16 | 377848 samples/sec | 0.34 |
|  |  |  |  | Ascend: 32 \* Ascend 910 </br> CPU：768 Cores | Mixed | 8000*32 | 433423 samples/sec | 0.20 |

1. 以上数据基于Atlas 800测试获得，且网络模型为模型并行。
2. 业界其他开源框架数据可参考：[Wide & Deep For TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep)。
