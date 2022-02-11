MindSpore Serving Documents
===========================

MindSpore Serving is a lightweight and high-performance service module that helps MindSpore developers efficiently deploy online inference services in the production environment. After completing model training on MindSpore, you can export the MindSpore model and use MindSpore Serving to create an inference service for the model.

MindSpore Serving provides the following functions:

- Customization of model preprocessing and postprocessing, simplifying model release and deployment
- The batch function used to split and combine requests containing multiple instances to meet the batch size requirements of the model
- Distributed model inference
- The gRPC APIs and easy-to-use Python encapsulation APIs on the client
- The RESTful APIs on the client

.. raw:: html

   <img src="https://gitee.com/mindspore/docs/raw/r1.6/docs/serving/docs/source_en/images/serving_en.png" width="700px" alt="" >

Typical MindSpore Serving Application Scenarios
------------------------------------------------

1. `Quick Start <https://www.mindspore.cn/serving/docs/en/r1.6/serving_example.html>`_

   Use a simple Add network as an example to describe how to use MindSpore Serving to deploy an inference service.

2. `Using the gRPC APIs to Access Services <https://www.mindspore.cn/serving/docs/en/r1.6/serving_grpc.html>`_

   Easily access services with high performance.

3. `Using the RESTful APIs to Access Services <https://www.mindspore.cn/serving/docs/en/r1.6/serving_restful.html>`_

   Access services based on HTTP.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Installation

   serving_install


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Guide

   serving_example
   serving_distributed_example
   serving_grpc
   serving_restful
   serving_model
   serving_multi_subgraphs

.. toctree::
   :maxdepth: 1
   :caption: API References

   server
   client

.. toctree::
   :maxdepth: 1
   :caption: References

   faq
