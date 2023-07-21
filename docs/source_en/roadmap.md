# RoadMap

MindSpore's top priority plans in the year are displayed as follows. We will continuously adjust the priority based on user feedback.

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.6/docs/source_en/roadmap.md)

In general, we will make continuous improvements in the following aspects:
1. Support more preset models.
2. Continuously supplement APIs and operator libraries to improve usability and programming experience.
3. Comprehensively support for Huawei Ascend AI processor and continuously optimize the performance and software architecture.
4. Improve visualization, debugging and optimization, and security-related tools.

We sincerely hope that you can join the discussion in the user community and contribute your suggestions.

## Preset Models
* CV: Classic models for object detection, GAN, image segmentation, and posture recognition.
* NLP: RNN and Transformer neural network, expanding the application based on the BERT pre-training model.
* Other: GNN, reinforcement learning, probabilistic programming, and AutoML.

## Usability
* Supplement APIs such as operators, optimizers, and loss functions.
* Complete the native expression support of the Python language.
* Support common Tensor/Math operations.
* Add more application scenarios of automatic parallelization to improve the accuracy of policy search.

## Performance Optimization
* Optimize the compilation time.
* Low-bit mixed precision training and inference.
* Improve memory utilization.
* Provide more fusion optimization methods.
* Improve the execution performance in PyNative.

## Architecture Evolution
* Optimize computational graph and operator fusion. Use fine-grained graph IR to express operators to form intermediate representation (IR) with operator boundaries and explore more layer optimization opportunities.
* Support more programming languages.
* Optimize the automatic scheduling and distributed training data cache mechanism of data augmentation.
* Continuously improve MindSpore IR.
* Support distributed training in parameter server mode.

## MindInsight Debugging and Optimization
* Training process observation
   * Histogram
   * Optimize the display of computational and data graphs.
   * Integrate the performance profiling and debugger tools.
   * Support comparison between multiple trainings.
* Training result lineage
   * Data augmentation lineage comparison.
* Training process diagnosis
   * Performance profiling.
   * Graph model-based debugger.

## MindArmour Security Hardening Package
* Test the model security.
* Provide model security hardening tools.
* Protect data privacy during training and inference.

## Inference Framework
* Support TensorFlow, Caffe, and ONNX model formats.
* Support iOS.
* Improve more CPU operators.
* Support more CV/NLP models.
* Online learning.
* Support deployment on IoT devices.
* Low-bit quantization.
* CPU and NPU heterogeneous scheduling.
