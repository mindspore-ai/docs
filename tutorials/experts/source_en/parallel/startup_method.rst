Distributed Parallel Startup Methods
====================================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.1/tutorials/experts/source_en/parallel/startup_method.rst

.. toctree::
  :maxdepth: 1
  :hidden:

  dynamic_cluster
  ms_operator

Multi-card Startup Method
-------------------------

Currently GPU, Ascend and CPU support multiple startup methods respectively, three of which are OpenMPI, dynamic networking and multi-process startup.

- Multi-process startup. The user needs to start the process corresponding to the number of cards, as well as configure the rank_table table. You can visit `Run Script <https://www.mindspore.cn/tutorials/experts/en/r2.1/parallel/train_ascend.html#running-the-script>`_ to learn how to start multi-card tasks by multi-processing.
- OpenMPI. The user can start running scripts via the mpirun command, at which point the user needs to provide the host file file. The user can visit `Running Scripts through OpenMPI <https://www.mindspore.cn/tutorials/experts/en/r2.1/parallel/train_ascend.html#running-the-script-through-openmpi>`_ to learn how to use OpenMPI to start a multi-card task.
- Dynamic cluster. MindSpore uses an internal dynamic networking module that does not require dependencies on external profiles or modules to help implement multi-card tasks. The user can visit `Dynamic Cluster <https://www.mindspore.cn/tutorials/experts/en/r2.1/parallel/dynamic_cluster.html>`_ to learn how to use dynamic networking way to start multi-card tasks.

+--------------------------+-------------------+-----------------+------------------+
|                          | GPU               | Ascend          | CPU              |
+==========================+===================+=================+==================+
| OpenMPI                  | Supported         | Supported       | Not supported    |
+--------------------------+-------------------+-----------------+------------------+
| Multi-process startup    | Not supported     | Supported       | Not supported    |
+--------------------------+-------------------+-----------------+------------------+
| Dynamic cluster          | Supported         | Supported       | Supported        |
+--------------------------+-------------------+-----------------+------------------+

Startup MindSpore Distributed Parallel Training on the Cloud
------------------------------------------------------------

MindSpore Operator is a MindSpore plugin for distributed training on Kubernetes.
The CRD (Custom Resource Definition) defines three roles: Scheduler, PS, and Worker, and users can easily implement distributed training by simply configuring the yaml file.

The current ms-operator supports normal single-Worker training, single-Worker training in PS mode, and Scheduler and Worker startup with automatic parallelism (e.g. data parallelism, model parallelism, etc.). Please refer to `ms-operator <https://gitee.com/mindspore/ms-operator>`_ for the detailed process.