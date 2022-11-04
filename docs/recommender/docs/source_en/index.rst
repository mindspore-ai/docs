MindSpore Recommender Documents
================================

MindSpore Recommender is an open source training acceleration library based on the MindSpore framework for the recommendation domain. With MindSpore's large-scale heterogeneous computing acceleration capability, MindSpore Recommender supports efficient training of large-scale dynamic features for online and offline scenarios.

.. raw:: html

   <p style="text-align: center;"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/recommender/docs/source_en/images/architecture.png" width="600px" alt="" ></p>

The MindSpore Recommender acceleration library consists of the following components:

- online training: implements online training of real-time data and incremental model updates by streaming data from real-time data sources (e.g., Kafka) and online real-time data processing to support business scenarios that require real-time model updates.
- offline training: for the traditional offline dataset training scenario, it supports the training of recommendation models containing large-scale feature vectors through automatic parallelism, distributed feature caching, heterogeneous acceleration and other technical solutions.
- data processing: MindPandas and MindData provide the ability to read and process data online and offline, saving the overhead of multiple languages and frameworks through full-Python expression support, and opening up efficient data flow links for data processing and model training.
- model library: includes continuous rich training of typical recommendation models. After rigorous validation for accuracy and performance, it can be used right after installation.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Installation

   install

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Guide

   offline_learning
   online_learning

.. toctree::
   :maxdepth: 1
   :caption: API References

   recommender