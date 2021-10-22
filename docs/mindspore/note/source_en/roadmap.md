# RoadMap

<!-- TOC -->

- [RoadMap](#roadmap)
    - [Preset Models](#preset-models)
    - [Usability](#usability)
    - [Performance Optimization](#performance-optimization)
    - [Architecture Evolution](#architecture-evolution)
    - [MindInsight Debugging and Optimization](#mindinsight-debugging-and-optimization)
    - [MindArmour Security Hardening Package](#mindarmour-security-hardening-package)
    - [Inference Framework](#inference-framework)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/note/source_en/roadmap.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

MindSpore's top priority plans in the year are displayed as follows. We will continuously adjust the priority based on user feedback.

In general, we will make continuous improvements in the following aspects:

1. Support more preset models.
2. Continuously supplement APIs and operator libraries to improve usability and programming experience.
3. Comprehensively support for Huawei Ascend AI processor and continuously optimize the performance and software architecture.
4. Improve visualization, debugging and optimization, and security-related tools.

We sincerely hope that you can join the discussion in the user community and contribute your suggestions.

## Preset Models

- CV: Classic models for object detection, GAN, image segmentation, and posture recognition.
- NLP: RNN and Transformer neural network, expanding the application based on the BERT pre-training model.
- Other: GNN, reinforcement learning, probabilistic programming, and AutoML.

## Usability

- Supplement APIs such as operators, optimizers, and loss functions.
- Complete the native expression support of the Python language.
- Support common Tensor/Math operations.
- Add more application scenarios of automatic parallelization to improve the accuracy of policy search.

## Performance Optimization

- Optimize the compilation time.
- Low-bit mixed precision training and inference.
- Improve memory utilization.
- Provide more fusion optimization methods.
- Improve the execution performance in PyNative.

## Architecture Evolution

- Optimize computational graph and operator fusion. Use fine-grained graph IR to express operators to form intermediate representation (IR) with operator boundaries and explore more layer optimization opportunities.
- Support more programming languages.
- Optimize the automatic scheduling and distributed training data cache mechanism of data augmentation.
- Continuously improve MindSpore IR.
- Support distributed training in parameter server mode.

## MindInsight Debugging and Optimization

- Training process observation
    - Histogram
    - Optimize the display of computational and data graphs.
    - Integrate the performance profiling and debugger tools.
    - Support comparison between multiple trainings.
- Training result lineage
    - Data augmentation lineage comparison.
- Training process diagnosis
    - Performance profiling.
    - Graph model-based debugger.

## MindArmour Security Hardening Package

- Test the model security.
- Provide model security hardening tools.
- Protect data privacy during training and inference.

## Inference Framework

- Continuous optimization for operator, and add more operator.
- Support NLP neural networks.
- Visualization for MindSpore lite model.
- MindSpore Micro, which supports ARM Cortex-A and Cortex-M with Ultra-lightweight.
- Support re-training and federated learning on mobile device.
- Support auto-parallel.
- MindData on mobile device, which supports image resize and pixel data transform.
- Support post-training quantize, which supports inference with mixed precision to improve performance.
- Support Kirin NPU, MTK APU.
- Support inference for multi models with pipeline.
- C++ API for model construction.
