.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore Federated Documents
================================

MindSpore Federated is a cryptographically distributed machine learning technique for solving data silos and performing efficient, secure and reliable machine learning across multiple parties or multiple resource computing nodes. Support the various participants of machine learning to build AI models together without directly sharing local data, including but not limited to mainstream deep learning models such as ad recommendation, classification, and detection, mainly applied in finance, medical, recommendation and other fields.

MindSpore Federated is an open source federation learning framework that provides a horizontal federated model with sample federation and a vertical federation model with feature federation. Support commercial deployment for millions of stateless terminal devices, as well as cloud federated between data centers across trusted zones, enabling full-scene intelligent applications without taking user data out of the local context.

Advantages of the MindSpore Federated Horizontal Framework
-----------------------------------------------------------

Horizontal Federated Architecture:

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/federated/docs/source_en/images/HFL_en.png" width="700px" alt="" >

1. Privacy Protection

   It supports accuracy-lossless security aggregation solution based on secure multi-party computation (MPC) to prevent model theft.

   It supports performance-lossless encryption based on local differential privacy to prevent private data leakage from models.

   It supports a gradient protection scheme based on Symbolic Dimensional Selection (SignDS), which prevents model privacy data leakage while reducing communication overhead by 99%.

2. Distributed Federated Aggregation

   The loosely coupled cluster processing mode on the cloud and distributed gradient quadratic aggregation paradigms support the deployment of tens of millions of heterogeneous devices, implements high-performance and high-availability federated aggregation computing, and can cope with network instability and sudden load changes.

3. Federated Learning Efficiency Improvement

   The adaptive frequency modulation strategy and gradient compression algorithm are supported to improve the federated learning efficiency and saving bandwidth resources.

   Multiple federated aggregation policies are supported to improve the smoothness of federated learning convergence and optimize both global and local accuracies.

4. Easy to Use

   Only one line of code is required to switch between the standalone training and federated learning modes.

   The network models, aggregation algorithms, and security algorithms are programmable, and the security level can be customized.

   It supports the effectiveness evaluation of federated training models and provides monitoring capabilities for federated tasks.

Advantages of the MindSpore Federated Vertical Framework
-----------------------------------------------------------

Vertical Federated Architecture:

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/federated/docs/source_en/images/VFL_en.png" width="700px" alt="" >

1. Privacy Protection

   Support high-performance Privacy Set Intersection Protocol (PSI), which prevents federated participants from obtaining ID information outside the intersection and can cope with data imbalance scenarios.

   Support encryption scheme of Label based on differential privacy to prevent leakage of user label data.

   Support privacy protection scheme combined software with hardware, which can reduce the number of communication rounds and communication volume compared with homomorphic encryption method.

2. Federated training

   Support multiple types of split learning network structures.

   Cross-domain training for large models with pipelined parallel optimization.

MindSpore Federated Working Process
------------------------------------

1. `Scenario Identification and Data Accumulation <https://www.mindspore.cn/federated/docs/en/r2.0.0-alpha/image_classification_application.html#preparation>`_

   Identify scenarios where federated learning is used and accumulate local data for federated tasks on the client.

2. `Model Selection and Framework Deployment <https://www.mindspore.cn/federated/docs/en/r2.0.0-alpha/image_classification_application.html#generating-a-device-side-model-file>`_

   Select or develop a model prototype and use a tool to generate a federated learning model that is easy to deploy.

3. `Application Deployment <https://www.mindspore.cn/federated/docs/en/r2.0.0-alpha/image_classification_application.html#simulating-multi-client-participation-in-federated-learning>`_

   Deploy the corresponding components to the business application and set up federated configuration tasks and deployment scripts on the server.

Common Application Scenarios
----------------------------

1. `Image Classification <https://www.mindspore.cn/federated/docs/en/r2.0.0-alpha/image_classification_application.html>`_

   Use the federated learning to implement image classification applications.

2. `Text Classification <https://www.mindspore.cn/federated/docs/en/r2.0.0-alpha/sentiment_classification_application.html>`_

   Use the federated learning to implement text classification applications.

.. toctree::
   :maxdepth: 1
   :caption: Deployment

   federated_install
   deploy_federated_server
   deploy_federated_client
   deploy_vfl

.. toctree::
   :maxdepth: 1
   :caption: Horizontal Application

   image_classfication_dataset_process
   image_classification_application
   sentiment_classification_application
   image_classification_application_in_cross_silo
   object_detection_application_in_cross_silo

.. toctree::
   :maxdepth: 1
   :caption: Vertical Application

   data_join
   split_wnd_application
   split_pangu_alpha_application

.. toctree::
   :maxdepth: 1
   :caption: Security and Privacy

   local_differential_privacy_training_noise
   local_differential_privacy_training_signds
   pairwise_encryption_training
   private_set_intersection
   secure_vertical_federated_learning_with_TEE
   secure_vertical_federated_learning_with_DP

.. toctree::
   :maxdepth: 1
   :caption: Communication Compression

   communication_compression

.. toctree::
   :maxdepth: 1
   :caption: API References

   Horizontal_FL_Server
   Horizontal_FL_Client
   Vertical_FL_Server
   Data_Join

.. toctree::
   :maxdepth: 1
   :caption: References

   faq