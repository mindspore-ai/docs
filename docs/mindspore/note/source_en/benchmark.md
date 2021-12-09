# Benchmarks

<!-- TOC -->

- [Benchmarks](#benchmarks)
    - [Training Performance](#training-performance)
        - [ResNet](#resnet)
        - [BERT](#bert)
        - [Wide & Deep (data parallel)](#wide--deep-data-parallel)
        - [Wide & Deep (Host-Device model parallel)](#wide--deep-host-device-model-parallel)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/note/source_en/benchmark.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

This document describes the MindSpore benchmarks.
For details about the MindSpore networks, see [ModelZoo](https://gitee.com/mindspore/models/tree/master).

## Training Performance

### ResNet

| Network | Network Type | Dataset | MindSpore Version | Resource &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Precision | Batch Size | Throughput | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 v1.5 | CNN | ImageNet2012 | 0.5.0-beta | Ascend: 1 * Ascend 910 </br> CPU: 24 Cores | Mixed | 256 | 2115 images/sec | - |
|  |  |  |  | Ascend: 8 * Ascend 910 </br> CPU: 192 Cores | Mixed | 256 | 16600 images/sec | 0.98 |
|  |  |  |  | Ascend: 16 * Ascend 910 </br> CPU: 384 Cores | Mixed | 256 | 32768 images/sec | 0.96 |

1. The preceding performance is obtained based on ModelArts, the HUAWEI CLOUD AI development platform. It is the average performance obtained by the Ascend 910 AI processor during the overall training process.
2. For details about other open source frameworks, see [ResNet-50 v1.5 for TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5).

### BERT

| Network | Network Type | Dataset | MindSpore Version | Resource &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Precision | Batch Size | Throughput |  Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BERT-Large | Attention | zhwiki | 0.5.0-beta | Ascend: 1 * Ascend 910 </br> CPU: 24 Cores | Mixed | 96 | 269 sentences/sec | - |
|  |  |  |  | Ascend: 8 * Ascend 910 </br> CPU: 192 Cores | Mixed | 96 | 2069 sentences/sec | 0.96 |

1. The preceding performance is obtained based on ModelArts, the HUAWEI CLOUD AI development platform. The network contains 24 hidden layers, the sequence length is 128 tokens, and the vocabulary contains 21128 tokens.
2. For details about other open source frameworks, see [BERT For TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT).

### Wide & Deep (data parallel)

| Network | Network Type | Dataset | MindSpore Version | Resource &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Precision | Batch Size | Throughput |  Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wide & Deep | Recommend | Criteo | 0.6.0-beta | Ascend: 1 * Ascend 910 </br> CPU: 24 Cores | Mixed | 16000 | 796892 samples/sec | - |
|  |  |  |  | Ascend: 8 \* Ascend 910 </br> CPU: 192 Cores | Mixed | 16000*8 | 4872849 samples/sec | 0.76 |

1. The preceding performance is obtained based on Atlas 800, and the model is data parallel.
2. For details about other open source frameworks, see [Wide & Deep For TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep).

### Wide & Deep (Host-Device model parallel)

| Network | Network Type | Dataset | MindSpore Version | Resource &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Precision | Batch Size | Throughput |  Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wide & Deep | Recommend | Criteo | 0.6.0-beta | Ascend: 1 * Ascend 910 </br> CPU: 24 Cores | Mixed | 1000 | 68715 samples/sec | - |
|  |  |  |  | Ascend: 8 \* Ascend 910 </br> CPU: 192 Cores | Mixed | 8000*8 | 283830 samples/sec | 0.51 |
|  |  |  |  | Ascend: 16 \* Ascend 910 </br> CPU: 384 Cores | Mixed | 8000*16 | 377848 samples/sec | 0.34 |
|  |  |  |  | Ascend: 32 \* Ascend 910 </br> CPU: 768 Cores | Mixed | 8000*32 | 433423 samples/sec | 0.20 |

1. The preceding performance is obtained based on Atlas 800, and the model is model parallel.
2. For details about other open source frameworks, see [Wide & Deep For TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep).
