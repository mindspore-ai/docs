MindSpore Serving Documents
===========================

MindSpore Serving is a lightweight and high-performance module that helps developers efficiently deploy inference services in production. Simply train your model on MindSpore, export it and then use MindSpore Serving to create inference services for the models.

MindSpore Serving supports:

- Customized preprocessing and postprocessing to simplify model release and deployment
- Batch function that splits and combines multiple-instances requests to fit the batch size requirements of the model
- Distributed model inference
- gRPC APIs and easy-to-use Python encapsulation APIs on the client
- RESTful APIs on the client

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/serving/docs/source_en/images/serving_en.png" width="700px" alt="" >

Typical Application Scenarios
-------------------------------

1. `Quick Start <https://www.mindspore.cn/serving/docs/en/master/serving_example.html>`_

   Use the Add network as an example to demonstrate how to deploy an inference service with MindSpore Serving.

2. `Access Services with gRPC APIs <https://www.mindspore.cn/serving/docs/en/master/serving_grpc.html>`_

   Easily access services with high performance.

3. `Using the RESTful APIs to Access Services <https://www.mindspore.cn/serving/docs/en/master/serving_restful.html>`_

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

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE