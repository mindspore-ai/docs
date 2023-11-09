Distributed Parallel Startup Methods
====================================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_en/parallel/startup_method.rst

.. toctree::
  :maxdepth: 1
  :hidden:

  dynamic_cluster
  mpirun
  rank_table

Startup Method
---------------

Currently GPU, Ascend and CPU support multiple startup methods respectively, three of which are dynamic cluster, \ ``mpirun`` and \ ``rank table``:

- `Dynamic cluster <https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/dynamic_cluster.html>`_: this method does not rely on third-party libraries, has disaster recovery function, good security, and supports three hardware platforms. It is recommended that users prioritize the use of this startup method.
- `mpirun <https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/mpirun.html>`_: this method relies on the open source library OpenMPI, and startup command is simple. Multi-machine need to ensure two-by-two password-free login. It is recommended for users who have experience in using OpenMPI to use this startup method.
- Dynamic cluster. MindSpore uses an internal dynamic networking module that does not require dependencies on external profiles or modules to help implement multi-card tasks. The user can visit `Dynamic Cluster <https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/dynamic_cluster.html>`_ to learn how to use dynamic networking way to start multi-card tasks.
- `rank table <https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/rank_table.html>`_: this method requires the Ascend hardware platform and does not rely on third-party library. After manually configuring the rank_table file, you can start the parallel program via a script, and the script is consistent across multiple machines for easy batch deployment.

The hardware support for the three startup methods is shown in the table below:

+-------------------------+--------------+-----------------+-------------+
|                         | GPU          | Ascend          | CPU         |
+=========================+==============+=================+=============+
|    Dynamic cluster      | Support      | Support         | Support     |
+-------------------------+--------------+-----------------+-------------+
|  \ ``mpirun``\          | Support      | Support         | Not support |
+-------------------------+--------------+-----------------+-------------+
|    \ ``rank table``\    | Not support  | Support         | Not support |
+-------------------------+--------------+-----------------+-------------+