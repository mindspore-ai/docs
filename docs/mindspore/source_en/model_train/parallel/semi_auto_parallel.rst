Semi-automatic Parallel
===========================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source_en.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindspore/source_en/model_train/parallel/semi_auto_parallel.rst
    :alt: View Source on Gitee

.. toctree::
  :maxdepth: 1
  :hidden:

  operator_parallel
  advanced_operator_parallel
  optimizer_parallel
  pipeline_parallel

Semi-automatic parallel supports the automatic mixing of multiple parallel modes, including:

- `Operator-level parallel <https://www.mindspore.cn/docs/en/r2.4.0/model_train/parallel/operator_parallel.html>`_: Operator-level parallel refers to slicing the input tensor and model parameters into multiple devices for computation on an operator basis to improve overall speed.
- `Higher-order Operator-level Parallelism <https://www.mindspore.cn/docs/en/r2.4.0/model_train/parallel/advanced_operator_parallel.html>`_: Higher-order operator-level parallelism refers to operator-level parallelism that allows customized device layout with tensor layout for more complex sharding logic.
- `Optimizer  parallel <https://www.mindspore.cn/docs/en/r2.4.0/model_train/parallel/optimizer_parallel.html>`_: Optimizer parallel reduces redundant computations on multiple devices for the same weight updates, spreading the computation over multiple devices.
- `Pipeline parallel <https://www.mindspore.cn/docs/en/r2.4.0/model_train/parallel/pipeline_parallel.html>`_: Pipeline parallel means that the model is sliced by layer, with each device processing only a certain part of the model.
