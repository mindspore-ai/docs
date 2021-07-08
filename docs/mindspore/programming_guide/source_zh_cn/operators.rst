算子
===========

概述
-----------

MindSpore的算子组件，可从算子使用方式和算子功能两种维度进行划分。以下示例代码需在PyNative模式运行。

算子使用方式
----------------

算子相关接口主要包括operations、functional和composite，可通过ops直接获取到这三类算子。

- operations提供单个的Primitive算子。一个算子对应一个原语，是最小的执行对象，需要实例化之后使用。

- composite提供一些预定义的组合算子，以及复杂的涉及图变换的算子，如`GradOperation`。

- functional提供operations和composite实例化后的对象，简化算子的调用流程。

mindspore.ops.operations
^^^^^^^^^^^^^^^^^^^^^^^^

operations提供了所有的Primitive算子接口，是开放给用户的最低阶算子接口。算子支持情况可查询[算子支持列表](https://www.mindspore.cn/docs/note/zh-CN/r1.3/operator_list.html)。

Primitive算子也称为算子原语，它直接封装了底层的Ascend、GPU、AICPU、CPU等多种算子的具体实现，为用户提供基础算子能力。

Primitive算子接口是构建高阶接口、自动微分、网络模型等能力的基础。

代码样例如下：

.. code-block::

    import numpy as np
    import mindspore
    from mindspore import Tensor
    import mindspore.ops.operations as P

    input_x = mindspore.Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
    input_y = 3.0
    pow = P.Pow()
    output = pow(input_x, input_y)
    print("output =", output)

运行结果如下：

.. code-block::

    output = [ 1.  8. 64.]

mindspore.ops.functional
^^^^^^^^^^^^^^^^^^^^^^^^

为了简化没有属性的算子的调用流程，MindSpore提供了一些算子的functional版本。入参要求参考原算子的输入输出要求。算子支持情况可以查询[算子支持列表](https://www.mindspore.cn/docs/note/zh-CN/r1.3/operator_list_ms.html#mindspore-ops-functional)。

例如`P.Pow`算子，我们提供了functional版本的`F.tensor_pow`算子。

使用functional的代码样例如下：

.. code-block::

    import numpy as np
    import mindspore
    from mindspore import Tensor
    from mindspore.ops import functional as F

    input_x = mindspore.Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
    input_y = 3.0
    output = F.tensor_pow(input_x, input_y)
    print("output =", output)

运行结果如下：

.. code-block::

    output = [ 1.  8. 64.]

mindspore.ops.composite
^^^^^^^^^^^^^^^^^^^^^^^

composite提供了一些算子的组合，包括`clip_by_value`和`random`相关的一些算子，以及涉及图变换的函数（`GradOperation`、`HyperMap`和`Map`等）。

算子的组合可以直接像一般函数一样使用，例如使用`normal`生成一个随机分布：

.. code-block::

    from mindspore import dtype as mstype
    from mindspore.ops import composite as C
    from mindspore import Tensor

    mean = Tensor(1.0, mstype.float32)
    stddev = Tensor(1.0, mstype.float32)
    output = C.normal((2, 3), mean, stddev, seed=5)
    print("output =", output)

运行结果如下：

.. code-block::

    output = [[2.4911082 0.7941146 1.3117087]
    [0.3058231 1.7729738 1.525996 ]]

以上代码运行于MindSpore的GPU版本。

operations/functional/composite三类算子合并用法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为了在使用过程中更加简便，除了以上介绍的几种用法外，我们还将`operations`，`functional`和`composite`三种算子封装到了`mindspore.ops`中，推荐直接调用`mindspore.ops`下的接口。

代码样例如下：

.. code-block::

  import mindspore.ops.operations as P
  pow = P.Pow()

.. code-block::

  import mindspore.ops as ops
  pow = ops.Pow()

以上两种写法效果相同。

.. toctree::
  :maxdepth: 1
  :hidden:

  operator_functions
  custom_operator