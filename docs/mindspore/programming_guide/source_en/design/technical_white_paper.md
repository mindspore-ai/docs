# Technical White Paper

`Ascend` `GPU` `CPU` `Design`

<!-- TOC -->

- [Technical White Paper](#technical-white-paper)
    - [Introduction](#introduction)
    - [Overview](#overview)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/design/technical_white_paper.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Introduction

Deep learning research and application have experienced explosive development in recent decades, triggering the third wave of artificial intelligence and achieving great success in image recognition, speech recognition and synthesis, unmanned driving, and machine vision. This also poses higher requirements on the algorithm application and dependent frameworks. With the continuous development of deep learning frameworks, a large quantity of computing resources can be conveniently used when neural network models are trained on large datasets.

Deep learning is a kind of machine learning algorithm that uses a multi-layer structure to automatically learn and extract high-level features from raw data. Generally, it is very difficult to extract high-level abstract features from raw data. There are two mainstream deep learning frameworks. One is to build a static graph before execution to define all operations and network structures, for example, TensorFlow. This method improves the training performance at the cost of usability. The other is dynamic graph computing that is executed immediately, for example, PyTorch. Different from static graphs, dynamic graphs are more flexible and easier to debug, but the performance is sacrificed. Therefore, the existing deep learning framework cannot meet the requirements of easy development and efficient execution at the same time.

## Overview

MindSpore is a next-generation deep learning framework that incorporates the best practices of the industry. It best manifests the computing power of the Ascend AI Processor and supports flexible all-scenario deployment across device-edge-cloud. MindSpore creates a brand-new AI programming paradigm and lowers the threshold for AI development. MindSpore aims to achieve easy development, efficient execution, and all-scenario coverage. To facilitate easy development, MindSpore adopts an automatic differentiation (AD) mechanism based on source code transformation (SCT), which can represent complex combinations through control flows. A function is converted into an intermediate representation (IR) which constructs a computational graph that can be parsed and executed on devices. Before execution, multiple software and hardware collaborative optimization technologies are used in the graph to improve performance and efficiency in various scenarios across the device, edge, and cloud. MindSpore supports dynamic graphs for checking the running mode. Thanks to the AD mechanism, the mode switching between dynamic and static graphs becomes very simple. To effectively train large models on large datasets, MindSpore supports data parallel, model parallel, and hybrid parallel training through advanced manual configuration policies, which is highly flexible. In addition, MindSpore supports the automatic parallelism which efficiently searches for a fast parallel strategy in a large strategy space. For details about the advantages of the MindSpore framework,

see [Technical White Paper](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com:443/white_paper/MindSpore_white_paper_enV1.1.pdf).
