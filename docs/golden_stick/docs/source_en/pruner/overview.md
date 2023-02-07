# Pruning Algorithm Overview

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/golden_stick/docs/source_en/pruner/overview.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

The following describes some basic concepts of pruning algorithms to help users understand the pruning algorithms. If you have a deep understanding of pruning algorithms, skip to [Examples](#examples).

## What Is Pruning

Just like neural networks, the pruning algorithm is inspired by neurobiology. In neurobiology, there is a concept called synaptic pruning, which usually occurs among infants of mammals. Through the synaptic pruning mechanism, neuron connections in the brain are simplified and reconstructed so that the brain can work more efficiently with lower energy. Pruning is to reduce network parameters by removing some components (such as a weight, a feature map, and a convolution kernel) in a neural network while ensuring that network accuracy is just slightly reduced, so as to reduce storage and computing costs during network deployment.

Neural network inference can be regarded as the process of activation and weight computation. Therefore, pruning is classified into activation pruning and weight pruning. In the MindSpore Golden Stick, we only discuss weight pruning.

Weight pruning is classified into structured pruning mode and unstructured pruning mode. As shown in the figure, from left to right, there are three types of pruning method: neuron pruning, channel pruning and filter pruning. The difference between them is the granularity of their pruning: a single neuron, a single channel or the entire filter, respectively. It can be seen that the pruning granularity of these three types of pruning is getting coarser and coarser, and the corresponding weights obtained after pruning are becoming more and more structured:

![Pruning modes](../images/pruner/pruner.png)

Usually we take neuron pruning as unstructured pruning which can prune the weight at any position in the weight with a single weight as the granularity. This pruning mode has less impact on the accuracy of the network due to its fine granularity, but it leads to the sparseness of the weight tensor. The sparse weight tensor is not friendly to memory access and parallel computing. Therefore, it is difficult for the unstructured pruned network to obtain a high acceleration ratio.

Channel pruning and filter pruning are generally considered to be structured pruning which can prune the weight of the model with the weighted channel or the entire convolution kernel as the granularity. Because an entire channel or an entire convolution kernel is directly pruned, a weight obtained through pruning is more regular and has a smaller scale. This is the meaning of structured pruning. Compared with unstructured pruning, structured pruning obtains more regular weights and is more friendly to memory access. Therefore, structured pruning is suitable for accelerated inference on devices such as the CPU and GPU.

## Examples

In the current version, the MindSpore Golden Stick provides a structured weight pruning algorithm. For details, see [SCOP Algorithm Example](https://www.mindspore.cn/golden_stick/docs/en/r0.3.0-alpha/pruner/scop.html).
