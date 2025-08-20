# Overall Structure

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/introduction/overview.md)

## Overview

The overall architecture of MindSpore Transformers is as follows:

![/overall_architecture](./images/overall_architecture.png)

The northbound API of MindSpore Transformers supports users integrating into their own training and inference platforms or open-source components, supporting Ascend's own technology stack while also actively embracing the open-source community, as follows:

1. Training platforms: MindCluster, third-party platforms  
2. Service components: vLLM  
3. Communities: Modelers, Hugging Face

MindSpore Transformers Southbound is based on MindSpore+Ascend's large-scale model technology stack, leveraging the MindSpore framework combined with CANN to optimize Ascend hardware for compatibility, providing a high-performance model training and inference experience.

MindSpore Transformers is mainly divided into the following modules:

1. Large model training and inference unified scheduling API: Provides a unified launcher script msrun_launcher.sh to uniformly execute the distributed training and inference processes of all models within the scheduling suite.
2. Registration/configuration layer: Implements a factory class by interface type to enable the high-level interface layer to initialize the corresponding task interface and model interface according to the configuration.
3. Large model library: Implements a high-performance large model library and basic Transformer interfaces, supporting both user-configurable custom model construction and custom development to accommodate various development scenarios.
4. Dataset: Implements data loading encapsulation for large model training and fine-tuning tasks, natively supporting Hugging Face Datasets, Megatron datasets, and MindSpore's native MindRecord data support.
5. Training Components: Implements the basic interfaces for the training process, including learning rate strategies, optimizers, training callbacks, and TrainOneStepWrapper interfaces.
6. Tool Layer: Independent tool scripts currently provide data preprocessing, Hugging Face weight conversion, and benchmarking tool scripts.
7. DFX (Design for X): Implements high-availability features such as fault diagnosis and fault monitoring to reduce the cost of recovering from training failures.

## Model Architecture

MindSpore Transformers has adopted a brand-new model architecture in version 1.6.0 and later. In the previous architecture (labeled as Legacy), each model had its own set of model code, making maintenance and optimization challenging. The new architecture (labeled as Mcore) employs a layered abstraction and modular implementation for large-scale general-purpose Transformer architectures, encompassing lower-level foundational layers such as Linear, Embedding, and Norm, as well as upper-level components like MoELayer, TransformerBlock, and the unified model interface GPTModel (General PreTrained Model). All modular interfaces are deeply optimized for parallelism leveraging MindSpore’s parallel computing capabilities, providing high-performance, out-of-the-box interfaces. All highly encapsulated and integrated interfaces support flexible combination through the ModuleSpec mechanism for model construction.

## Training Capabilities

MindSpore Transformer training offers a range of efficient and user-friendly features, as well as ecosystem collaboration capabilities, to assist users in achieving simplicity, efficiency, and stability during the pre-training and fine-tuning phases of large models. External capabilities include:

- Multi-dimensional hybrid parallelism, including data parallelism, model parallelism, optimizer parallelism, pipeline parallelism, sequence parallelism, context parallelism, and MoE expert parallelism;
- Support for directly loading Megatron-LM multi-source mixed datasets during the pre-training phase, avoiding data migration issues across platforms and frameworks;
- In the fine-tuning phase, it integrates Hugging Face ecosystem capabilities, supporting the use of Hugging Face SFT datasets, Hugging Face Tokenizer for data preprocessing, reading Hugging Face model configurations to instantiate models, and loading native Hugging Face Safetensors weights. Combined with zero-code, configuration-enabled low-parameter fine-tuning capabilities, it achieves efficient and convenient fine-tuning;
- Supports automatic weight splitting and loading in distributed environments, eliminating the need for manual weight conversion during distributed strategy switching debugging, cluster scaling, and other scenarios, thereby facilitating efficient debugging and training;
- Provides user-friendly and highly available features such as training status monitoring, fault recovery, anomaly skipping, and resume training from breakpoints, supporting testability, maintainability, and reliability during pre-training/fine-tuning processes;
- Encapsulates high-performance basic interfaces, with interface design aligned with Megatron-LM and computational accuracy meeting standards. Combined with tutorials and documentation related to model migration and accuracy comparison, as well as the Cell-level dump tool provided by the Ascend toolchain, it achieves low-threshold, high-efficiency model migration and construction.

## Inference Capabilities

MindSpore Transformers inference integrates with third-party open-source components, providing developers with richer inference deployment, quantization, and evaluation capabilities:

- Supports direct loading and use of Hugging Face open-source configurations, weights, and tokenizers, enabling one-click inference startup;
- Supports integration with vLLM service frameworks, enabling service-based inference deployment. Supports service-based features such as Continuous Batch, Prefix Cache, and Chunked Prefill;
- Through the MindSpore Golden-Stick quantization suite, Legacy models can achieve A16W8, A8W8, and A8W4 quantization inference, while Mcore models are expected to support A8W8 and A8W4 quantization inference in the next version;
- Through the AISbench evaluation suite, MindSpore Transformers models integrated with vLLM service-oriented architecture can achieve CEval, GSM8K, AIME, and other 20+ mainstream benchmark evaluations.

The Southbound API of MindSpore Transformers relies on the inference optimization capabilities provided by the MindSpore framework to achieve high-performance inference in southbound:

- Relying on the multi-level pipeline dispatch feature provided by the framework Runtime, the operator scheduling is split into three tasks—InferShape, Resize, and Launch—on the host side for pipeline dispatch, fully utilizing the host's multi-threading resources to improve operator dispatch efficiency and achieve inference acceleration;
- By default, it uses the PyNative programming mode + JIT (just-in-time) compilation technology to compile the model into a static computation graph for inference acceleration. It can also be switched to the PyNative dynamic graph mode with a single click for convenient development and debugging;
- MindSpore Transformers supports the use of ACLNN, ATB, and MindSpore-provided inference acceleration/fusion operators to achieve more efficient inference performance on the Ascend platform.