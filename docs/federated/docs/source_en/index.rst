.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore Federated Documents
================================

MindSpore Federated is an open-source federated learning framework that supports the commercial deployment of tens of millions of stateless devices. It enables all-scenario intelligent applications when user data is stored locally.

The federated learning is an encrypted distributed machine learning technology that allows users participating in federated learning to build AI models without sharing local data. MindSpore Federated currently focuses on the large-scale participants in the horizontal federated learning scenarios.

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/federated/docs/source_en/r1.7/federated_en.png" width="700px" alt="" >

Advantages of the MindSpore Federated
---------------------------------------

1. Privacy Protection

   The MindSpore Federated framework data is stored locally for training. The data itself is not exchanged. Instead, the updated model parameters are exchanged in encryption mode.

   It supports accuracy-lossless security aggregation solution based on secure multi-party computation (MPC) to prevent model theft.

   It supports performance-lossless encryption based on local differential privacy to prevent private data leakage from models.

2. Distributed Federated Aggregation

   The loosely coupled cluster processing mode on the cloud supports the deployment of tens of millions of heterogeneous devices, implements high-performance and high-availability distributed federated aggregation computing, and can cope with network instability and sudden load changes.

3. Federated Learning Efficiency Improvement

   The synchronous and asynchronous federation modes and multiple model compression algorithms are supported to improve the federated learning efficiency and saving bandwidth resources.

   Multiple federated aggregation policies are supported to improve the smoothness of federated learning convergence and optimize both global and local accuracies.

4. Easy to Use

   Only one line of code is required to switch between the standalone training and federated learning modes.

   The network models, aggregation algorithms, and security algorithms are programmable, and the security level can be customized.

MindSpore Federated Working Process
------------------------------------

1. `Scenario Identification and Data Accumulation <https://www.mindspore.cn/federated/docs/en/master/image_classification_application.html#data-processing>`_

   Identify scenarios where federated learning is used and accumulate local data for federated tasks on the client.

2. `Model Selection and Client Deployment <https://www.mindspore.cn/federated/docs/en/r1.7/image_classification_application.html#generating-a-device-model-file>`_

   Select or develop a model prototype and use a tool to generate a device model that is easy to deploy.

3. `Application Deployment <https://www.mindspore.cn/federated/docs/en/r1.7/image_classification_application.html#simulating-multi-client-participation-in-federated-learning>`_

   Deploy the Federated-Client to the device application, and set the federated configuration task and deployment script on the cloud.

Common Application Scenarios
----------------------------

1. `Image Classification <https://www.mindspore.cn/federated/docs/en/r1.7/image_classification_application.html>`_

   Use the federated learning to implement image classification applications.

2. `Text Classification <https://www.mindspore.cn/federated/docs/en/r1.7/sentiment_classification_application.html>`_

   Use the federated learning to implement text classification applications.

.. toctree::
   :maxdepth: 1
   :caption: Deployment

   federated_install
   deploy_federated_server
   deploy_federated_client

.. toctree::
   :maxdepth: 1
   :caption: Application

   image_classification_application
   sentiment_classification_application

.. toctree::
   :maxdepth: 1
   :caption: Security and Privacy

   local_differential_privacy_training_noise
   pairwise_encryption_training

.. toctree::
   :maxdepth: 1
   :caption: API References

   federated_server
   federated_client

.. toctree::
   :maxdepth: 1
   :caption: References

   faq