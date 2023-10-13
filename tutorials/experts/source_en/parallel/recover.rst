Fault Recovery
==============

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.2/tutorials/experts/source_en/parallel/recover.rst

.. toctree::
  :maxdepth: 1
  :hidden:

  disaster_recover
  fault_recover

During the distributed parallel training process, MindSpore has three recovery methods when encountering problems such as failures of compute nodes or communication interruptions:

- `Model Reloading <https://www.mindspore.cn/tutorials/experts/en/r2.2/parallel/model_loading.html>`_: During training, by configuring the parameters to be merged and saved, a complete model parameter file is saved for each card, which can be directly loaded for checkpoint recovery.
- `Disaster Recovery in Dynamic Cluster Scenarios <https://www.mindspore.cn/tutorials/experts/en/r2.2/parallel/disaster_recover.html>`_: In the dynamic cluster startup scenario, if a process fails, the other processes will enter a waiting state, and the training task can be continued by pulling up the failed process without restarting the cluster (currently only supports GPU hardware platforms).
- `Fault Recovery Based on Redundant Information <https://www.mindspore.cn/tutorials/experts/en/r2.2/parallel/fault_recover.html>`_: In large model training, the devices divided according to the dimension of data parallelism have the same parameters of their models. According to this principle, these redundant parameter information can be utilized as a backup, and in case of one node failure, another node utilizing the same parameters can recover the failed node.