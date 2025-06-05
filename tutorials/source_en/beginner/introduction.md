[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/beginner/introduction.md)

**Introduction** || [Quick Start](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/quick_start.html#) || [Tensor](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/tensor.html) || [Data Loading and Processing](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/dataset.html) || [Model](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/autograd.html) || [Train](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/train.html) || [Save and Load](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/save_load.html) || [Accelerating with Static Graphs](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/accelerate_with_static_graph.html) || [Mixed Precision](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/mixed_precision.html)

# Overview

The following describes the Huawei AI full-stack solution and the position of MindSpore in the solution. Developers who are interested in MindSpore can visit the [MindSpore community](https://gitee.com/mindspore/mindspore) and click [Watch, Star, and Fork](https://gitee.com/mindspore/mindspore).

## Introduction to MindSpore

### Overall Architecture

The overall architecture of MindSpore is as follows:

1. Model Suite: Provides developers with ready-to-use models and development kits, such as the large model suite MindSpore Transformers, MindSpore ONE, and scientific computing libraries for hot research areas;
2. Deep Learning + Scientific Computing: Provides developers with various Python interfaces required for AI model development, maximizing compatibility with developers' habits in the Python ecosystem;
3. Core: As the core of the AI framework, it builds the Tensor data structure, basic operation operators, autograd module for automatic differentiation, Parallel module for parallel computing, compile capabilities, and runtime management module.

![arch](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/docs/mindspore/source_en/design/images/arch_en.png)

### Design Philosophy

MindSpore is a full-scenario deep learning framework designed to achieve three major goals: easy development, efficient execution, and unified deployment across all scenarios. Easy development is reflected in API friendliness and low debugging difficulty; efficient execution includes computational efficiency, data preprocessing efficiency, and distributed training efficiency; full-scenario means the framework simultaneously supports cloud, edge, and device-side scenarios.

## Introduction to Huawei Ascend AI Full-Stack Solution

Ascend computing is a full-stack AI computing infrastructure and application based on the Ascend series processors. It includes the Ascend series chips, Atlas series hardware, CANN chip enablement, MindSpore AI framework, ModelArts, and MindX application enablement.

Huawei Atlas AI computing solution is based on Ascend series AI processors and uses various product forms such as modules, cards, edge stations, servers, and clusters to build an all-scenario AI infrastructure solution oriented to device, edge, and cloud. It covers data center and intelligent edge solutions, as well as the entire inference and training processes in the deep learning field.

Th Ascend AI full stack is shown below:

![Ascend full stack](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/tutorials/source_en/beginner/images/introduction1.png)

The functions of each module are described as follows:

- **Ascend Application Enablement**: AI platform or service capabilities provided by Huawei major product lines based on MindSpore.
- **MindSpore**: Support for device-edge-cloud-independent and collaborative unified training and inference frameworks.
- **CANN**: A driver layer that enables Ascend chips ([learn more](https://www.hiascend.com/en/software/cann)).
- **Compute Resources**: Ascend serialized IP, chips and servers.

For details, click [Huawei Ascend official website](https://e.huawei.com/en/products/servers/ascend).

## Joining the Community

Welcome every developer to the MindSpore community and contribute to this all-scenario AI framework.

- **MindSpore official website**: provides comprehensive MindSpore information, including installation, tutorials, documents, community, resources, and news ([learn more](https://www.mindspore.cn/en)).
- **MindSpore code**:

    - [MindSpore Gitee](https://gitee.com/mindspore/mindspore): Top 1 Gitee open-source project in 2020, where you can track the latest progress of MindSpore by clicking Watch, Star, and Fork, discuss issues, and commit code.

    - [MindSpore GitHub](https://github.com/mindspore-ai/mindspore): MindSpore code image of Gitee. Developers who are accustomed to using GitHub can learn MindSpore and view the latest code implementation here.

- **MindSpore forum**: We are dedicated to serving every developer. You can find your voice in MindSpore, regardless of whether you are an entry-level developer or a master. Let's learn and grow together. ([Learn more](https://discuss.mindspore.cn/))
