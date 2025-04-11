Distributed Parallel Startup Methods
====================================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source_en.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_en/parallel/startup_method.rst
    :alt: View Source On Gitee

.. toctree::
  :maxdepth: 1
  :hidden:

  msrun_launcher
  dynamic_cluster
  mpirun
  rank_table

Startup Method
---------------

Currently GPU, Ascend and CPU support multiple startup methods respectively, four of which are \ ``msrun``, dynamic cluster, \ ``mpirun`` and \ ``rank table``:

- `msrun <https://www.mindspore.cn/tutorials/en/r2.6.0/parallel/msrun_launcher.html>`_: `msrun` is the capsulation of Dynamic cluster. It allows user to launch distributed jobs using one single command in each node. It could be used after MindSpore is installed. This method does not rely on third-party libraries and configuration files, has disaster recovery function, good security, and supports three hardware platforms. It is recommended that users prioritize the use of this startup method.
- `Dynamic cluster <https://www.mindspore.cn/tutorials/en/r2.6.0/parallel/dynamic_cluster.html>`_: dynamic cluster requires user to spawn multiple processes and export environment variables. It's the implementation of `msrun`. Use this method when running `Parameter Server` training mode. For other distributed jobs, `msrun` is recommended.
- `mpirun <https://www.mindspore.cn/tutorials/en/r2.6.0/parallel/mpirun.html>`_: this method relies on the open source library OpenMPI, and startup command is simple. Multi-machine need to ensure two-by-two password-free login. It is recommended for users who have experience in using OpenMPI to use this startup method.
- `rank table <https://www.mindspore.cn/tutorials/en/r2.6.0/parallel/rank_table.html>`_: this method requires the Ascend hardware platform and does not rely on third-party library. After manually configuring the rank_table file, you can start the parallel program via a script, and the script is consistent across multiple machines for easy batch deployment.

The hardware support for the four startup methods is shown in the table below:

+-------------------------+--------------+-----------------+-------------+
|                         | GPU          | Ascend          | CPU         |
+=========================+==============+=================+=============+
|    \ ``msrun``\         | Support      | Support         | Support     |
+-------------------------+--------------+-----------------+-------------+
|    Dynamic cluster      | Support      | Support         | Support     |
+-------------------------+--------------+-----------------+-------------+
|  \ ``mpirun``\          | Support      | Support         | Not support |
+-------------------------+--------------+-----------------+-------------+
|    \ ``rank table``\    | Not support  | Support         | Not support |
+-------------------------+--------------+-----------------+-------------+