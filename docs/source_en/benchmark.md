# Benchmarks

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.3/docs/source_en/benchmark.md)

This document describes the MindSpore benchmarks. 
For details about the MindSpore pre-trained model, see [Model Zoo](https://gitee.com/mindspore/mindspore/tree/r0.3/mindspore/model_zoo).

## Training Performance

### ResNet

| Network |     Network Type | Dataset | MindSpore Version | Resource &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Precision | Batch Size | Throughput | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 v1.5 | CNN | ImageNet2012 | 0.2.0-alpha | Ascend: 1 * Ascend 910 </br> CPU：24 Cores | Mixed | 32 | 1787 images/sec | - |
|  |  |  |  | Ascend: 8 * Ascend 910 </br> CPU：192 Cores | Mixed | 32 | 13689 images/sec | 0.95 |
|  |  |  |  | Ascend: 16 * Ascend 910 </br> CPU：384 Cores | Mixed | 32 | 27090 images/sec | 0.94 |

1. The preceding performance is obtained based on ModelArts, the HUAWEI CLOUD AI development platform. It is the average performance obtained by the Ascend 910 AI processor during the overall training process. 
2. For details about other open source frameworks, see [ResNet-50 v1.5 for TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/RN50v1.5#nvidia-dgx-2-16x-v100-32g).

### BERT

| Network |	Network Type | Dataset | MindSpore Version | Resource &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Precision | Batch Size | Throughput |  Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BERT-Large | Attention | zhwiki | 0.2.0-alpha | Ascend: 1 * Ascend 910 </br> CPU：24 Cores | Mixed | 96 | 210 sentences/sec | - |
|  |  |  |  | Ascend: 8 * Ascend 910 </br> CPU：192 Cores | Mixed | 96 | 1613 sentences/sec | 0.96 |

1. The preceding performance is obtained based on ModelArts, the HUAWEI CLOUD AI development platform. The network contains 24 hidden layers, the sequence length is 128 tokens, and the vocabulary contains 21128 tokens.   
2. For details about other open source frameworks, see [BERT For TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT).