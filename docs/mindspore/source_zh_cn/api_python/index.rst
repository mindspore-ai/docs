API 文档
=========================

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   mindspore
   mindspore.device_context
   mindspore.nn
   mindspore.ops
   mindspore.ops.primitive
   mindspore.mint
   mindspore.amp
   mindspore.train
   mindspore.communication
   mindspore.communication.comm_func
   mindspore.common.initializer
   mindspore.runtime
   mindspore.dataset
   mindspore.nn.probability
   mindspore.rewrite
   mindspore.multiprocessing
   mindspore.boost
   mindspore.numpy
   mindspore.scipy
   mindspore.utils
   mindspore.hal
   mindspore.experimental
   env_var_list
   ../note/api_mapping/pytorch_api_mapping

MindSpore提供了丰富的模型构建、训练、推理等接口，各模块接口功能说明如下。

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - 模块名
     - 模块说明
   * - `mindspore <./mindspore.html>`_
     - 框架基础接口。
   * - `mindspore.nn <./mindspore.nn.html>`_
     - 神经网络层，用于构建神经网络中的预定义构建块或计算单元。
   * - `mindspore.ops <./mindspore.ops.html>`_
     - 函数接口。
   * - `mindspore.ops.primitive <./mindspore.ops.primitive.html>`_
     - Primitive的算子。
   * - `mindspore.mint <./mindspore.mint.html>`_
     - 与业界主流用法一致的functional、nn、优化器接口。
   * - `mindspore.amp <./mindspore.amp.html>`_
     - 混合精度接口。
   * - `mindspore.train <./mindspore.train.html>`_
     - 训练接口。
   * - `mindspore.communication <./mindspore.communication.html>`_
     - 集合通信接口。
   * - `mindspore.communication.comm_func <./mindspore.communication.comm_func.html>`_
     - 集合通信函数式接口。
   * - `mindspore.common.initializer <./mindspore.common.initializer.html>`_
     - 参数初始化。
   * - `mindspore.hal <./mindspore.hal.html>`_
     - 设备管理、流管理、事件管理与内存管理的接口。
   * - `mindspore.dataset <./mindspore.dataset.loading.html>`_
     - 加载和处理各种数据集的接口。
   * - `mindspore.dataset.transforms <./mindspore.dataset.transforms.html>`_
     - 通用数据变换。
   * - `mindspore.mindrecord <./mindspore.mindrecord.html>`_
     - MindSpore开发的高效数据格式MindRecord相关的操作接口。
   * - `mindspore.nn.probability <./mindspore.nn.probability.html>`_
     - 可参数化的概率分布和采样函数。
   * - `mindspore.rewrite <./mindspore.rewrite.html>`_
     - 基于自定义规则的模型源码修改接口。
   * - `mindspore.multiprocessing <./mindspore.multiprocessing.html>`_
     - 多进程接口。
   * - `mindspore.boost <./mindspore.boost.html>`_
     - 自动加速网络接口。
   * - `mindspore.numpy <./mindspore.numpy.html>`_
     - 类NumPy接口。
   * - `mindspore.scipy <./mindspore.scipy.html>`_
     - 类SciPy接口。
   * - `mindspore.utils <./mindspore.utils.html>`_
     - 工具接口。
   * - `mindspore.experimental <./mindspore.experimental.html>`_
     - 实验性接口。
   * - `环境变量 <./env_var_list.html>`_
     - 环境变量相关说明。
