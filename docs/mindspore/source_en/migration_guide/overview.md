# Overview

<a href="https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_en/migration_guide/overview.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source_en.png"></a>

This migration guide contains the complete steps for migrating neural networks to MindSpore from other machine learning frameworks, mainly PyTorch.

The following figure shows the migration process.:

1. Configure the MindSpore development environment
2. Analyze the network model to be migrated and acquire basic data
3. MindSpore reproduction. It is recommended to use PYNATIVE mode to debug the model during the functional debugging stage and switch to GRAPH mode after the functional debugging is completed. After the model development is completed, it is recommended to reproduce the inference process first and the training process later.
4. Debugging and tuning for function, precision and performance.

In this process, we have a relatively complete description of each link. We hope that through the migration guide, developers can quickly migrate the existing code of other frameworks to MindSpore.

![flowchart](./images/flowchart.PNG "Migration Process")

## [Environmental Preparation and Information Acquisition](https://www.mindspore.cn/docs/en/r1.10/migration_guide/enveriment_preparation.html)

Network migration starts with configuring the MindSpore development environment, and this chapter describes the installation process and knowledge preparation in detail. The knowledge preparation includes a basic introduction to the MindSpore components models and hub, including the purpose, scenarios and usage. There are also tutorials on training on the cloud: using ModelArts to adapt scripts, uploading datasets in OBS, and training online.

## [Model analysis and preparation](https://www.mindspore.cn/docs/en/r1.10/migration_guide/analysis_and_preparation.html)

Before doing formal development, some analysis preparation work needs to be done on the network/algorithm to be migrated, including:

- Reading papers and reference codes to understand algorithms and network structures
- Reproducing the results of the paper, obtaining the base model (ckpt), benchmark accuracy and performance
- Analyzing the APIs and functions used in the network.

When migrating networks from PyTorch to MindSpore, users need to be aware of [differences from typical PyTorch interfaces](https://www.mindspore.cn/docs/en/r1.10/migration_guide/typical_api_comparision.html).

## [MindSpore model implementation](https://www.mindspore.cn/docs/en/r1.10/migration_guide/model_development/model_development.html)

After the preliminary analysis preparation, you can develop the new network by using MindSpore. This chapter will introduce the knowledge of MindSpore network construction and the process of training and inference, starting from the basic modules during inference and training, and using one or two examples to illustrate how to build the network in special scenarios.

## [Debugging and Tuning](https://www.mindspore.cn/docs/en/r1.10/migration_guide/debug_and_tune.html)

This chapter will introduce some methods of debugging and tuning from three aspects: function, precision and performance.

## [Example of Network Migration Debugging](https://www.mindspore.cn/docs/en/r1.10/migration_guide/sample_code.html)

This chapter contains a complete network migration sample. From the analysis and replication of the benchmark network, it details the steps of script development and precision debugging and tuning, and finally lists the common problems and corresponding optimization methods during the migration process, framework performance issues.

## [FAQs](https://www.mindspore.cn/docs/en/r1.10/migration_guide/faq.html)

This chapter lists the frequently-asked questions and corresponding solutions.
