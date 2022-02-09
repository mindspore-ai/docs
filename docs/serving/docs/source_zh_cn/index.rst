MindSpore Serving 文档
=========================

MindSpore Serving是一个轻量级、高性能的服务模块，旨在帮助昇思MindSpore开发者在生产环境中高效部署在线推理服务。当用户使用昇思MindSpore完成模型训练后，导出MindSpore模型，即可使用MindSpore Serving创建该模型的推理服务。

MindSpore Serving包含以下功能：

- 支持自定义关于模型的预处理和后处理，简化模型的发布和部署。
- 支持batch功能，包含多个实例的请求会被拆分组合以满足模型batch size的需要。
- 支持分布式模型推理功能。
- 支持客户端gRPC接口，提供简单易用的客户端Python封装接口。
- 支持客户端RESTful接口。

.. raw:: html

   <img src="https://gitee.com/mindspore/docs/raw/master/docs/serving/docs/source_zh_cn/images/serving_cn.png" width="700px" alt="" >

使用MindSpore Serving的典型场景
--------------------------------

1. `快速入门 <https://www.mindspore.cn/serving/docs/zh-CN/master/serving_example.html>`_

   以一个简单的Add网络为例，演示如何使用MindSpore Serving部署推理服务。

2. `使用gRPC接口访问服务 <https://www.mindspore.cn/serving/docs/zh-CN/master/serving_grpc.html>`_

   高性能、简单方便地访问服务。

3. `使用RESTful接口访问服务 <https://www.mindspore.cn/serving/docs/zh-CN/master/serving_restful.html>`_

   基于HTTP协议访问服务。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装部署

   serving_install



.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 使用指南

   serving_example
   serving_distributed_example
   serving_grpc
   serving_restful
   serving_model
   serving_multi_subgraphs

.. toctree::
   :maxdepth: 1
   :caption: API参考

   server
   client

.. toctree::
   :maxdepth: 1
   :caption: 参考文档

   faq
