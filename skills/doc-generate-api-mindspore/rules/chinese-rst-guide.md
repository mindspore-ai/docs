# Chinese RST Guide (中文RST编写指南)

## File Structure (文件结构)

Every RST file must start with a title line followed by `=` underline, then a `.. py::` directive:

- **Title line** = full dotted path (identical to filename and directive path)
- **Underline**: `=` characters, length at least as long as the title
- **Blank line** before the directive

```rst
full.path.ClassName.method_name
===============================

.. py:method:: full.path.ClassName.method_name()
```

## Opening Description (开头描述)

- One-line summary; use "逐元素" for elementwise operations
- Include purpose/usage
- Include math formula with `.. math::` if applicable
- Add variable definitions for the formula
- Description text must NOT contain colons (`：` or `:`)

## Args (参数)

- Format: `- **name** (Type) - Description.`
- Indent: secondary params align with the `*` of primary params
- Optional: `(类型, 可选)` — add `可选` after type
- Default: `默认值： ``xxx`` 。`
- Dtype lists use Chinese commas: `float16、float32、float64、bfloat16`
- Parameter order must match the API definition exactly

## Keyword Args (关键字参数)

- Place after `参数：` when parameters follow a `*` separator
- Use `关键字参数：` heading
- Same format as Args

## Returns (返回)

- Description must NOT contain colons — use commas instead
- Format: `Tensor, shape和数据类型与输入相同。`

## Raises (异常)

- Format: `- **ValueError** - `x` 的shape长度小于1。`
- Document all explicitly raised exceptions (except TypeError)
- Merge same-type exceptions for readability

## Inputs / Outputs (输入 / 输出)

- Used for classes and Cell-like constructs instead of Args/Returns
- Inputs: `- **x** - shape为 :math:`(*, in\_features)` 的Tensor。`
- Outputs: Description of the output tensor

---

## Reference (参考)

### Format Examples (格式示例)

#### Function RST

```rst
mindspore.nn.polynomial_decay_lr
================================

.. py:function:: mindspore.nn.polynomial_decay_lr(learning_rate, end_learning_rate, total_step, step_per_epoch, decay_epoch, power, update_decay_epoch=False)

    基于多项式衰减函数计算学习率。每个step的学习率将会被存放在一个列表中。

    对于第i步，计算decayed_learning_rate[i]的公式为：

    .. math::
        decayed\_learning\_rate[i] = (learning\_rate - end\_learning\_rate) *
        (1 - tmp\_epoch / tmp\_decay\_epoch)^{power} + end\_learning\_rate

    其中，

    .. math::
        tmp\_epoch = \min(current\_epoch, decay\_epoch)

    .. math::
        current\_epoch=floor(\frac{i}{step\_per\_epoch})

    .. math::
        tmp\_decay\_epoch = decay\_epoch

    如果 `update_decay_epoch` 为 ``True`` ，则每个epoch更新 :math:`tmp\_decay\_epoch` 的值。公式为：

    .. math::
        tmp\_decay\_epoch = decay\_epoch * ceil(current\_epoch / decay\_epoch)

    参数：
        - **learning_rate** (float) - 学习率的初始值。
        - **end_learning_rate** (float) - 学习率的最终值。
        - **total_step** (int) - step总数。
        - **step_per_epoch** (int) - 每个epoch的step数。
        - **decay_epoch** (int) - 进行衰减的epoch数。
        - **power** (float) - 多项式的幂，必须大于0。
        - **update_decay_epoch** (bool，可选) - 如果为 ``True`` ，则更新 `decay_epoch` 。默认值： ``False`` 。

    返回：
        list[float]。列表的大小为 `total_step`。

    异常：
        - **ValueError** - `learning_rate` 或 `power` 小于等于0。
```

#### Class RST

```rst
mindspore.nn.Dense
==================

.. py:class:: mindspore.nn.Dense(in_channels, out_channels, weight_init=None, bias_init=None, has_bias=True, activation=None, dtype=mstype.float32)

    全连接层，对输入执行密集连接操作。

    .. math::
        \text{outputs} = \text{activation}(\text{X} * \text{kernel} + \text{bias})

    参数：
        - **in_channels** (int) - Dense层输入Tensor的空间维度。
        - **out_channels** (int) - Dense层输出Tensor的空间维度。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]，可选) - 权重参数的初始化方法。默认值： ``None`` ，权重使用HeUniform初始化。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]，可选) - 偏置参数的初始化方法。默认值： ``None`` ，偏差使用Uniform初始化。
        - **has_bias** (bool，可选) - 是否使用偏置向量。默认值： ``True`` 。
        - **activation** (Union[str, Cell, Primitive, None]，可选) - 应用于全连接层输出的激活函数。默认值： ``None`` 。
        - **dtype** (:class:`mindspore.dtype`，可选) - Parameter的数据类型。默认值： ``mstype.float32`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(*, in\_channels)` 的Tensor。

    输出：
        shape为 :math:`(*, out\_channels)` 的Tensor。

    异常：
        - **ValueError** - `weight_init` 的shape长度不等于2。
```

#### Method RST

```rst
mindspore.nn.Cell.insert_param_to_cell
======================================

.. py:method:: mindspore.nn.Cell.insert_param_to_cell(param_name, param, check_name_contain_dot=True)

    将指定名称的参数添加到Cell中。

    参数：
        - **param_name** (str) - 参数名称。
        - **param** (Parameter) - 要插入到Cell的参数。
        - **check_name_contain_dot** (bool，可选) - 是否对 `param_name` 中的"."进行检查。默认值： ``True`` 。

    异常：
        - **KeyError** - 如果参数名称为空或包含"."。
```

### RST Formatting Conventions (RST格式规范)

#### Definition Directives

| Type | Format |
| ------ | -------- |
| Function | `.. py:function:: full.path.name(param1, param2)` |
| Class | `.. py:class:: full.path.ClassName(param1, param2)` |
| Method | `.. py:method:: method_name(param1)` (4sp indent from class), optionally add `:staticmethod:` / `:classmethod:` / `:abstractmethod:` on a new line (4sp indent) |
| Overloaded function | Add `:noindex:` on a new line, indented 4 spaces |

#### Special Directives

| Directive | Format |
| ----------- | -------- |
| Note | `.. note::` + 4-space indent |
| Warning | `.. warning::` + 4-space indent |
| See Also | `.. seealso::` + 4-space indent |
| Math block | `.. math::` + indented formula |
| Inline math | `` :math:`formula` `` |
| Code block | `.. code-block::` + blank line + indented code |

#### Parameter Indentation

```text
参数：
    - **param1** (Tensor) - Description line 1.
    - **param2** (tuple[int]) - Description.

      - **sub_param** (int) - Indent aligns with `*` above.
```

#### Dimensions

- 零维 (0-D), 一维 (1-D), 二维 (2-D), 三维 (3-D), 四维 (4-D)

#### Cross-references

| Reference | Format |
| ----------- | -------- |
| Class | `` :class:`mindspore.nn.Cell` `` |
| Function | `` :func:`mindspore.ops.rsqrt` `` |
| Special method | `` :meth:`__getitem__` `` (from :class:`mindspore.dataset.dataloader.Dataset`) |
| External link | `` `链接文本 <https://xxx>`_ `` |

#### Images

- `.. image:: ../../images/Foo.png` (path relative to RST file)

#### Tables

Use `.. list-table::` directive. See Sphinx docs for details.

#### Special Characters

| Character | Escaped form |
| ----------- | ------------- |
| `*` (emphasis) | `\*` |
| `**` (bold) | `\*\*` |
| `:` (in descriptions) | Avoid entirely — use commas |
| Backtick (code) | ````text```` |

#### Things Not to Include

- ❌ `样例：` / Examples section (auto-generated)
- ❌ `支持平台：` / Supported Platforms (auto-generated)
- ❌ Colons in description text or return value text

#### Math Notation

```rst
Inline: :math:`x^2 + y^2`

Block:
.. math::

    \text{out}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
```

Common patterns:

- Element-wise: `out_{i} = f(input_{i})`
- Matrix: `Y = xA^T + b`
- Reduction: `out = \sum_{i} input_{i}`

#### dtype Documentation

```rst
# Float ops:
# 支持数据类型：float16、float32、float64、bfloat16。
# Integer ops:
# 支持数据类型：int8、int16、int32、int64、uint8。
# All numeric:
# 支持数据类型：int8、int16、int32、int64、uint8、uint16、uint32、uint64、
#     float16、float32、float64、bfloat16。
# Complex:
# 支持数据类型：complex64、complex128。
```

Note: Chinese dtype lists use Chinese commas (、).

#### Shape Documentation

```rst
# - **input** - Shape：:math:`(N, C, H, W)`。
# - **x** - Shape：:math:`(*)`，其中 :math:`*` 表示任意数量的维度。
# - **y** - Shape：:math:`(*)`，必须与 `x` 可广播。
```
