# Operators Usage

<!-- TOC -->

- [Operators Usage](#operators-usage)
    - [Overview](#overview)
    - [mindspore.ops.operations](#mindsporeopsoperations)
    - [mindspore.ops.functional](#mindsporeopsfunctional)
    - [mindspore.ops.composite](#mindsporeopscomposite)
    - [Combination usage of operations, functional and composite three types of operators](#combination-usage-of-operations-functional-and-composite-three-types-of-operators)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/operators_usage.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

APIs related to operators include operations, functional, and composite. Operators related to these three APIs can be directly obtained using ops.

- The operations API provides a single primitive operator. An operator corresponds to a primitive and is the smallest execution object. An operator can be used only after being instantiated.
- The composite API provides some predefined composite operators and complex operators involving graph transformation, such as `GradOperation`.
- The functional API provides objects instantiated by the operations and composite to simplify the operator calling process.

## mindspore.ops.operations

The operations API provides all primitive operator APIs, which are the lowest-order operator APIs open to users. For details about the supported operators, see [Operator List](https://www.mindspore.cn/docs/note/en/master/operator_list.html).

Primitive operators directly encapsulate the implementation of operators at bottom layers such as Ascend, GPU, AICPU, and CPU, providing basic operator capabilities for users.

Primitive operator APIs are the basis for building high-order APIs, automatic differentiation, and network models.

A code example is as follows:

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops.operations as P

input_x = mindspore.Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
input_y = 3.0
pow = P.Pow()
output = pow(input_x, input_y)
print("output =", output)
```

The following information is displayed:

```text
output = [ 1.  8. 64.]
```

## mindspore.ops.functional

To simplify the calling process of operators without attributes, MindSpore provides the functional version of some operators. For details about the input parameter requirements, see the input and output requirements of the original operator. For details about the supported operators, see [Operator List](https://www.mindspore.cn/docs/note/en/master/operator_list_ms.html#mindspore-ops-functional).

For example, the functional version of the `P.Pow` operator is `F.tensor_pow`.

A code example is as follows:

```python
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.ops import functional as F

input_x = mindspore.Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
input_y = 3.0
output = F.tensor_pow(input_x, input_y)
print("output =", output)
```

The following information is displayed:

```text
output = [ 1.  8. 64.]
```

## mindspore.ops.composite

The composite API provides some operator combinations, including some operators related to clip_by_value and random, and functions (such as `GradOperation`, `HyperMap`, and `Map`) related to graph transformation.

The operator combination can be directly used as a common function. For example, use `normal` to generate a random distribution:

```python
from mindspore import dtype as mstype
from mindspore.ops import composite as C
from mindspore import Tensor

mean = Tensor(1.0, mstype.float32)
stddev = Tensor(1.0, mstype.float32)
output = C.normal((2, 3), mean, stddev, seed=5)
print("output =", output)
```

The following information is displayed:

```text
output = [[2.4911082  0.7941146  1.3117087]
 [0.30582333  1.772938  1.525996]]
```

> The preceding code runs on the GPU version of MindSpore.

## Combination usage of operations, functional and composite three types of operators

In order to make it easier to use, in addition to the several usages introduced above, we have encapsulated the three operators of operations/functional/composite into mindspore.ops. It is recommended to directly call the interface in mindspore.ops.

The code sample is as follows:

```python
import mindspore.ops.operations as P
pow = P.Pow()
```

```python
import mindspore.ops as ops
pow = ops.Pow()
```

> The above two methods have the same effect.
