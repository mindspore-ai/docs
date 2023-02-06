# Model Compression

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/experts/source_en/infer/model_compression.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore is a full-scenario AI framework. When the model is deployed to the end-side or other lightweight devices, there are various restrictions on the memory, power consumption, latency, etc. of the deployment, so the model often needs to be compressed before deployment.

MindSpore model compression is supported by [MindSpore Golden Stick](https://www.mindspore.cn/golden_stick/docs/en/r0.3.0-alpha/index.html). MindSpore Golden Stick is a set of model compression algorithms jointly designed and developed by Huawei Noah team and Huawei MindSpore team, providing a series of model compression algorithms for MindSpore, such as quantization and pruning. Details of MindSpore Golden Stick can be found in the [MindSpore Golden Stick Official Profile](https://www.mindspore.cn/golden_stick/docs/en/r0.3.0-alpha/index.html).

## Quantization Algorithm

### Concept

Quantization is a process of approximating floating-point (FP32) weights and inputs to a limited number (usually int8) of discrete values (generally int8) at a low inference accuracy loss. It uses a data type with fewer bits to approximately represent 32-bit floating-point data with a limited range. The input and output of the network are still floating-point data. In this way, the size of network and memory consumption are reduced, and the network inference speed is accelerated.

Currently, there are two types of quantization solutions in the industry: **quantization aware training** and **post-training quantization**.

(1) **Quantization aware training** requires training data and has better network accuracy. It is applicable to scenarios that have high requirements on the network compression rate and accuracy. The purpose is to reduce accuracy loss. The forward inference process in which the gradient is involved in network training enables the network to obtain a difference of quantization loss. The gradient update needs to be performed in a floating point. Therefore, the gradient is not involved in a backward propagation process.

(2) **Post-training quantization** is easy to use. Only a small amount of calibration data is required. It is applicable to scenarios that require high usability and lack of training resources.

### Quantization Examples

- [SimQAT algorithm](https://www.mindspore.cn/golden_stick/docs/en/r0.3.0-alpha/quantization/simqat.html): A basic quantization aware algorithm based on the fake quantization technology
- [SLB quantization algorithm](https://www.mindspore.cn/golden_stick/docs/en/r0.3.0-alpha/quantization/slb.html): A non-linear low-bit quantization aware algorithm

## Pruning Algorithm

### Concept

Pruning is to reduce network parameters by removing some components (such as a weight, a feature map, and a convolution kernel) in a neural network while ensuring that network accuracy is just slightly reduced, so as to reduce storage and computing costs during network deployment.

Neural network inference can be regarded as the process of activation and weight computation. Therefore, pruning is classified into activation pruning and weight pruning. In the MindSpore Golden Stick, we only discuss weight pruning.

**Weight pruning** is classified into structured pruning mode and unstructured pruning mode.

(1) Usually we take neuron pruning as **unstructured pruning** which can prune the weight at any position in the weight with a single weight as the granularity. This pruning mode has less impact on the accuracy of the network due to its fine granularity, but it leads to the sparseness of the weight tensor. The sparse weight tensor is not friendly to memory access and parallel computing. Therefore, it is difficult for the unstructured pruned network to obtain a high acceleration ratio.

(2) Channel pruning and filter pruning are generally considered to be **structured pruning** which can prune the weight of the model with the weighted channel or the entire convolution kernel as the granularity. Because an entire channel or an entire convolution kernel is directly pruned, a weight obtained through pruning is more regular and has a smaller scale. This is the meaning of structured pruning. Compared with unstructured pruning, structured pruning obtains more regular weights and is more friendly to memory access. Therefore, structured pruning is suitable for accelerated inference on devices such as the CPU and GPU.

### Pruning Examples

- [SCOP pruning algorithm](https://www.mindspore.cn/golden_stick/docs/en/r0.3.0-alpha/pruner/scop.html)：A structured weight pruning algorithm
