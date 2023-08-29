Semi-automatic Parallel
===========================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png
    :target: https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/semi_auto_parallel.rst

.. toctree::
  :maxdepth: 1
  :hidden:

  operator_parallel
  optimizer_parallel
  pipeline_parallel

----------------------

Semi-automatic parallel supports the automatic mixing of multiple parallel modes, including:

- `Operator-level parallel <https://www.mindspore.cn/tutorials/experts/en/master/parallel/operator_parallel.html>`_: Operator-level parallel refers to slicing the input tensor and model parameters into multiple devices for computation on an operator basis to improve overall speed.
- `Optimizer  parallel <https://www.mindspore.cn/tutorials/experts/en/master/parallel/optimizer_parallel.html>`_: Optimizer parallel reduces redundant computations on multiple devices for the same weight updates, spreading the computation over multiple devices.
- `Pipeline parallel <https://www.mindspore.cn/tutorials/experts/en/master/parallel/pipeline_parallel.html>`_: Pipeline parallel means that the model is sliced by layer, with each device processing only a certain part of the model.