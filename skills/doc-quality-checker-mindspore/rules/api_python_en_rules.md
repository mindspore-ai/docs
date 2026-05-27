# Python API英文注释检查规则

**说明**：本文件包含Python API英文注释的检查规则：

- **Python Docstring**：Python代码中的docstrings注释（对应中文RST文档）
- **YAML配置**：function_doc/method_doc/op_def目录下的YAML配置文件（对应中文RST文档）

## Python API注释检查规则

### 总体要求

Python API注释必须符合 Python `doctest` 规范要求，示例中的 `>>>`、`...` 和输出行的格式应能通过 `python -m doctest` 验证。

### 必须的注释项

| 编号 | 注释项 | 要求 | 缺失检查 |
|-----|-------|------|---------|
| I-PY-R-01 | Summary | 必须有，描述接口功能 | 无Summary描述 |
| I-PY-R-02 | Args | 如有参数必须写 | 有参数但无Args |
| I-PY-R-03 | Returns | 如有返回值必须写 | 有返回值但无Returns |
| I-PY-R-04 | Raises | 如有 `TypeError` 以外的异常必须写，`TypeError` 不需要写 | 有异常但无Raises |

### 格式检查

| 编号 | 检查项 | 正确格式 | 错误示例 |
|-----|-------|---------|---------|
| I-PY-F-01 | Summary | 内容中无冒号 `:` | Summary中有冒号 |
| I-PY-F-02 | 冒号格式 | 关键字（如Args、Returns等）后有冒号 `:` | 冒号缺失或多余 |
| I-PY-F-03 | 空行要求 | 不同类型内容间有空行，同类型内容间无空行 | 空行位置错误 |
| I-PY-F-04 | 缩进要求 | `Args`、`Raises` 内容换行缩进4空格 | 缩进不正确 |
| I-PY-F-05 | 参数类型空格 | 参数名和 `(` 之间有空格 | 缺少空格 |

### Args注释检查

| 编号 | 参数类型 | 正确写法 | 错误示例 |
|-----|---------|---------|---------|
| I-PY-A-01 | 基本数据类型 | `arg1 (int): Some description.` | 类型写法错误 |
| I-PY-A-02 | dtype | `mindspore.dtype` 或 `numpy.dtype` | 格式不正确 |
| I-PY-A-03 | Union类型 | `Union[Tensor, Number]` | 格式不正确 |
| I-PY-A-04 | list类型 | `list[str]` | 格式不正确 |
| I-PY-A-05 | 可选类型 | `(类型, optional)` | 格式不正确 |
| I-PY-A-06 | 二级参数 | 缩进4空格，继续使用 `-` | 格式不正确 |

### Returns注释检查

| 编号 | 检查项 | 要求 |
|-----|-------|------|
| I-PY-T-01 | 类型说明 | 必须说明返回值类型 |
| I-PY-T-02 | 维度变化 | 如有变化需说明与输入的关系 |
| I-PY-T-03 | 多返回值 | 分行写，无序列表方式 |
| I-PY-T-04 | 冒号限制 | Returns内容中不能包含冒号 |

### Examples注释检查

| 编号 | 检查项 | 正确格式 | 错误示例 |
|-----|-------|---------|---------|
| I-PY-E-01 | 代码行前缀 | `>>>` 用于代码，`...` 用于换行或空行 | 前缀错误 |
| I-PY-E-02 | 输出行 | 无前缀，直接输出结果 | 错误添加前缀 |
| I-PY-E-03 | import语句 | 每行一个import | 多个import在一行 |
| I-PY-E-04 | 可运行性 | 示例代码必须可执行 | 代码无法运行 |
| I-PY-E-05 | 运行结果 | 需给出实际输出结果 | 缺失输出 |

### Python Examples代码格式

| 编号 | 检查项 | 正确格式 | 错误示例 |
|-----|-------|---------|---------|
| I-PY-EF-01 | 代码行前缀 | `>>>` 用于代码行 | 使用 `>>` |
| I-PY-EF-02 | 续行前缀 | `...` 用于多行代码或空行 | 缺少续行符 |
| I-PY-EF-03 | 输出行 | 无前缀，直接输出结果 | 错误添加 `>>>` |
| I-PY-EF-04 | import语句 | 每个模块单独一行 | 多个import在一行 |
| I-PY-EF-05 | 类定义 | 使用 `...` 作为续行符 | 未使用续行符 |
| I-PY-EF-06 | 函数调用 | `>>>` 后跟代码 | 缺少空格 |
| I-PY-EF-07 | 空格要求 | `result = func()` | `result=func()` 缺少空格 |
| I-PY-EF-08 | None值 | 使用 `None` | 使用 `null` |
| I-PY-EF-09 | True/False | 使用大写 | 小写或中文 |
| I-PY-EF-10 | 字符串引号 | 单引号或双引号一致 | 混用引号 |
| I-PY-EF-11 | 数组格式 | `[1, 2, 3]` | 使用其他格式 |

**Python Examples正确格式示例：**

```python
class Net(nn.Cell):
    """
    Some description.

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> class Net(nn.Cell):
        ...     def __init__(self, dense_shape):
        ...         super(Net, self).__init__()
        ...         self.dense_shape = dense_shape
        ...     def construct(self, indices, values):
        ...         x = SparseTensor(indices, values, self.dense_shape)
        ...         return x.values, x.indices, x.dense_shape
        ...
        >>> indices = Tensor([[0, 1], [1, 2]])
        >>> values = Tensor([1, 2], dtype=ms.float32)
        >>> out = Net((3, 4))(indices, values)
        >>> print(out[0])
        [1. 2.]
    """
```

**Python Examples错误格式示例：**

```python
# 错误：缺少 >>> 前缀
def add(a, b):
    """
    Examples:
        result = add(1, 2)
        print(result)
    """

# 错误：续行符错用
def add(a, b):
    """
    Examples:
        >>> def add(a, b):
        >>> return a + b
    """

# 错误：缺少 ... 续行符
def add(a, b):
    """
    Examples:
        >>> def add(a, b):
        return a + b
    """

# 错误：输出行添加了 >>>
def add(a, b):
    """
    Examples:
        >>> result = add(1, 2)
        >>> print(result)
        >>> 3
    """

# 错误：变量赋值缺少空格
def add(a,b):
    """
    Examples:
        >>> result=add(1,2)
    """

# 错误：多个import在同一行
def add(a, b):
    """
    Examples:
        >>> import mindspore, numpy as np
    """
```

**Python Examples检查要点：**

1. 代码行必须以 `>>>` 开头
2. 多行代码（含类或函数定义、人为换行等）或空行以 `...` 开头
3. 输出结果行开头不需要加任何符号
4. import语句每行一个
5. 示例代码必须可运行
6. 需给出实际运行结果

### Inputs/Outputs检查（算子和Cell）

| 编号 | 检查项 | 要求 |
|-----|-------|------|
| I-PY-IO-01 | Tensor类型 | 必须描述shape，使用 `:math:` 格式 |
| I-PY-IO-02 | 格式 | 使用 `- **name** (Type) - Description` 格式 |
| I-PY-IO-03 | 数学公式 | 使用 `.. math::` 或 `:math:` 行内公式 |
| I-PY-IO-04 | Supported Platforms | 名称前后加 ``，多个用空格隔开 |

### 公式检查

| 编号 | 检查项 | 正确格式 | 错误示例 |
|-----|-------|---------|---------|
| I-PY-M-01 | 行公式 | `.. math::` + 公式内容 | 格式错误 |
| I-PY-M-02 | 行内公式 | `` :math:`formula` `` | 格式错误 |
| I-PY-M-03 | 下划线变量 | 使用 `{}` 包裹或 `\_` 转义 | 格式错误 |
| I-PY-M-04 | 斜体变量 | 变量直接写，小写希腊字母 | 格式错误 |
| I-PY-M-05 | 正体函数 | exp, log, sin, cos等 | 错误使用斜体 |

### 链接检查

| 编号 | 检查项 | 正确格式 | 错误示例 |
|-----|-------|---------|---------|
| I-PY-L-01 | 只显示标题 | `` `name`_`` + ``.. _`name`: https://xxx`` | 格式错误 |
| I-PY-L-02 | 简化写法 | `` `name <https://xxx>`_`` | 格式错误 |
| I-PY-L-03 | 换行缩进 | 链接文本换行需缩进 | 未缩进 |
| I-PY-L-04 | https空格 | https前需有空格 | 紧贴文字 |

### 表格检查

| 编号 | 检查项 | 正确格式 |
|-----|-------|---------|
| I-PY-TB-01 | list-table | 使用 `.. list-table::` 指令 |
| I-PY-TB-02 | 标题 | `.. list-table:: Title` |
| I-PY-TB-03 | 列宽 | `:widths:` 属性 |
| I-PY-TB-04 | 表头 | `* - Heading` 格式 |
| I-PY-TB-05 | 空单元格 | 使用 `-` 表示 |

### 引用检查

| 编号 | 引用类型 | 正确格式 | 错误示例 |
|-----|---------|---------|---------|
| I-PY-RF-01 | 角色引用（class/func） | `:class:`/`:func:` 后跟可解析的名称，完整或简短均可 | 缺少冒号，如 `class:` / `func:` |
| I-PY-RF-02 | 变量名 | 使用 \` 包裹 | 格式错误 |
| I-PY-RF-03 | 变量值 | 使用 \`\` 包裹 | 格式错误 |

### 废弃接口检查

| 编号 | 检查项 | 要求 |
|-----|-------|------|
| I-PY-D-01 | 警告说明 | 说明将废弃，建议使用的接口 |
| I-PY-D-02 | 支持平台 | 写上 `Deprecated` |
| I-PY-D-03 | 示例 | 包含完整使用示例 |

### 图片引用

| 编号 | 检查项 | 正确格式 | 错误示例 |
|-----|-------|---------|---------|
| I-PY-IG-01 | 格式 | `.. image:: {name.png}` | 格式错误 |
| I-PY-IG-02 | 路径 | 图片名称（相对于文档位置） | 路径错误 |
| I-PY-IG-03 | 提交位置 | 提交到 api_python 对应模块目录 | 位置错误 |

### 特殊格式检查

| 编号 | 检查项 | 正确写法 | 错误写法 |
|-----|-------|---------|---------|
| I-PY-S-01 | 反斜杠处理 | 头部改为 `r"""` | 未改头部 |
| I-PY-S-02 | 动词时态 | 统一用第一人称或第三人称 | 时态不一致 |
| I-PY-S-03 | Note | 只能是 `Note:`，不是 `Notes:` | 复数形式 |
| I-PY-S-04 | Examples | 只能是 `Examples:`，不是 `Example:` | 单数形式 |

---

## YAML API注释检查规则（function_doc / method_doc / op_def）

YAML文件用于定义MindSpore Python API文档，主要有三类来源：

| 来源 | 文件路径 | 用途 | 调用方式 |
|------|----------|------|---------|
| function_doc | `mindspore/ops/api_def/function_doc/` | 函数接口（mint/ops模块） | `mint.add(x, y)` 或 `ops.add(x, y)` |
| method_doc | `mindspore/ops/api_def/method_doc/` | Tensor实例方法 | `Tensor.add(x, y)` 或 `x.add(y)` |
| op_def/yaml/doc | `mindspore/ops/op_def/yaml/doc/` | mindspore.ops算子接口 | `mindspore.ops.gather(x, y, axis)` |

### 接口定义格式

| 编号 | YAML类型 | 接口签名格式 | 示例 |
|-----|---------|-------------|------|
| I-YA-D-01 | function_doc | `add(input, other, *, alpha=1) -> Tensor` | 有完整签名，包含关键字参数标记 |
| I-YA-D-02 | method_doc | `add(other) -> Tensor` | 无self参数，隐含self |
| I-YA-D-03 | op_def/yaml/doc | （无接口签名） | 只有description描述功能 |

**正确示例 - function_doc：**

```yaml
add:
  description: |
    add(input, other, *, alpha=1) -> Tensor

    Scales the `other` value by `alpha` and adds it to `input`.
```

**正确示例 - method_doc：**

```yaml
add:
  description: |
    add(other) -> Tensor

    Adds other value to `self` element-wise.

  .. method:: Tensor.add(other, *, alpha=1) -> Tensor
      :noindex:

    Adds scaled other value to `self`.
```

**正确示例 - op_def/yaml/doc：**

```yaml
gather:
  description: |
    Returns the slice of the input tensor corresponding to the elements of `input_indices`.
```

### 参数格式

| 编号 | 参数类型 | 正确格式 | 错误示例 |
|-----|---------|---------|---------|
| I-YA-P-01 | 位置参数 | `input (Tensor): 描述` | 缺少参数类型 |
| I-YA-P-02 | 可选参数 | `(类型, optional): 描述` 或 `(类型, optional): 描述 Default ``xxx`` .` | 格式错误 |
| I-YA-P-03 | 关键字参数 | `Keyword Args:` + 参数 | 与Args混用 |
| I-YA-P-04 | Union类型 | `Union[Tensor, Number]` | 格式错误 |
| I-YA-P-05 | list类型 | `list[int]` | 格式错误 |

### 必填字段

YAML文档包含以下主要字段：`description`（接口描述+签名）、`Args`/`Keyword Args`（参数说明）、`Returns`（返回值）、`Raises`（异常）、`Supported Platforms`（支持平台）、`Examples`（示例代码）以及可选的 `Warning`/`Note` 等标签。

| 编号 | 字段 | 要求 | 适用场景 |
|-----|-----|------|---------|
| I-YA-F-01 | description | 必须有，function_doc和method_doc需包含接口签名 | 必填 |
| I-YA-F-02 | Args | 如有位置参数必须写 | 可选 |
| I-YA-F-03 | Keyword Args | 如有关键字参数必须写（仅function_doc和method_doc） | 可选 |
| I-YA-F-04 | Returns | 如有返回值必须写 | 可选 |
| I-YA-F-05 | Raises | 如有 `TypeError` 以外的异常必须写，`TypeError` 不需要写 | 可选 |
| I-YA-F-06 | Supported Platforms | 必须列出支持的平台 | 算子/方法/函数必填 |
| I-YA-F-07 | Examples | 推荐提供可运行示例 | 建议 |

### YAML字段格式检查

| 编号 | 字段 | 正确格式 | 错误示例 |
|-----|-------|---------|---------|
| I-YA-FT-01 | description | `description: \|` + 多行内容 | 单行描述 |
| I-YA-FT-02 | Args/Keyword Args | 列表格式，缩进4空格 | 缩进错误 |
| I-YA-FT-03 | Returns | `Returns: type, description` 或 `Returns: type` | 缺少类型 |
| I-YA-FT-04 | Raises | `ValueError: 描述` 或 `RuntimeError: 描述` 等，`TypeError` 不需要写 | 格式错误 |
| I-YA-FT-05 | Examples | `Examples: \|` + `>>>` 代码 | 缺少代码前缀 |
| I-YA-FT-06 | Supported Platforms | ``Ascend`` ``GPU`` ``CPU`` | 缺少反引号 |

### 特殊标签（Warning/Note）

| 编号 | 标签 | 正确格式 | 位置 |
|-----|-----|---------|------|
| I-YA-W-01 | Warning | `.. warning::` + 内容 | description内 |
| I-YA-W-02 | Note | `.. note::` + 内容 | description内 |

### 返回值格式

| 编号 | 类型 | 正确格式 | 错误示例 |
|-----|-----|---------|---------|
| I-YA-T-01 | 单返回值 | `Tensor, 描述` | 缺少逗号 |
| I-YA-T-02 | 多返回值 | `tuple[Tensor], 描述` | 格式错误 |
| I-YA-T-03 | 无返回值 | （省略Returns字段） | 写了空Returns |
| I-YA-T-04 | 简洁返回 | `Returns: Tensor` | 仅用于op_def/yaml/doc |

### 数学公式格式

| 编号 | 公式类型 | 正确格式 | 错误示例 |
|-----|---------|---------|---------|
| I-YA-M-01 | 行公式 | `.. math::` + 公式内容（缩进） | 格式错误 |
| I-YA-M-02 | 行内公式 | `` :math:`formula` `` | 格式错误 |
| I-YA-M-03 | 下划线变量 | 使用 `{}` 包裹（如 `xxx_{yyy}`）或 `\_` 转义 | 格式错误 |
| I-YA-M-04 | 斜体变量 | 变量直接写，小写希腊字母 | 格式错误 |
| I-YA-M-05 | 正体函数 | exp, log, sin, cos, tanh等用正体 | 错误使用斜体 |

**正确示例：**

```yaml
description: |
  add(input, other, *, alpha=1) -> Tensor

  .. math::
      out_{i} = input_{i} + alpha \times other_{i}

  The shape is :math:`input\_params.shape[:axis] + input\_indices.shape`.
```

**错误示例：**

```yaml
# 错误：行公式未缩进
.. math::
out_i = input_i + other_i

# 错误：下划线变量未转义
output[i_j] = input[i_j]
```

### Examples代码格式

| 编号 | 检查项 | 正确格式 | 错误示例 |
|-----|-------|---------|---------|
| I-YA-E-01 | 代码行前缀 | `>>>` 用于代码行 | 缺少前缀 |
| I-YA-E-02 | 输出行 | 无前缀，直接输出结果 | 错误添加 `>>>` |
| I-YA-E-03 | 空行 | 使用 `...` | 直接空行 |
| I-YA-E-04 | import语句 | 每行一个import | 多个import在一行 |

**Examples调用方式区分：**

| YAML类型 | 正确示例 |
|---------|---------|
| function_doc | `>>> output = mint.add(x, y)` 或 `>>> output = ops.add(x, y)` |
| method_doc | `>>> output = Tensor.add(x, y)` 或 `>>> output = x.add(y)` |
| op_def/yaml/doc | `>>> output = mindspore.ops.gather(x, y, axis)` |

### 重载函数格式（仅method_doc）

| 编号 | 检查项 | 正确格式 | 错误示例 |
|-----|-------|---------|---------|
| I-YA-O-01 | 重载标记 | `.. method:: Tensor.xxx` + `:noindex:` | 缺少 `:noindex:` |
| I-YA-O-02 | 缩进 | 与正文缩进一致，不需要顶格书写 | 顶格书写 |
| I-YA-O-03 | 方法名 | 不带类名前缀，直接写接口名 | 写成 `mindspore.Tensor.xxx` |
| I-YA-O-04 | 位置 | 在主接口后面单独写 | 与主接口混在一起 |
| I-YA-O-05 | :noindex:缩进 | 换行再缩进4格添加标签 | 缩进位置错误 |

**正确示例：**

```yaml
add:
  description: |
    add(other) -> Tensor
    Adds other value to `self` element-wise.

  Args:
    other (Union[Tensor, number.Number, bool]): The other input.

  .. method:: Tensor.add(other, *, alpha=1) -> Tensor
      :noindex:

  Adds scaled other value to `self`.
```

**错误示例：**

```yaml
# 错误：顶格书写
.. method:: Tensor.add(other, alpha=1) -> Tensor
    :noindex:

# 错误：使用了mindspore.Tensor前缀
.. method:: mindspore.Tensor.add(other, alpha=1) -> Tensor
    :noindex:

# 错误：缺少 :noindex:
.. method:: Tensor.add(other, alpha=1) -> Tensor
```

**说明：** 首行注释为接口定义，格式为 `xxx(param1, param2)`，不需要加类名前缀Tensor，直接写接口名即可。

### 接口引用格式

| 编号 | 检查项 | 正确格式 | 错误示例 |
|-----|-------|---------|---------|
| I-YA-R-01 | 函数引用 | `:func:`接口名` ` | 使用 `:class:` |
| I-YA-R-02 | 类方法引用 | `:func:`Tensor.xxx` ` | 错误使用 |
| I-YA-R-03 | 完整路径 | 不需要加 `mindspore.` 前缀 | 多余前缀 |

### 图片引用格式

| 编号 | YAML类型 | 正确格式 | 相对路径说明 |
|-----|---------|---------|-------------|
| I-YA-IG-01 | function_doc | （一般无图片） | - |
| I-YA-IG-02 | method_doc | `.. image:: ../../images/xxx.png` | 相对于中文文档位置 |
| I-YA-IG-03 | op_def/yaml/doc | `.. image:: ../images/xxx.png` | 相对于yaml文件位置 |

**正确示例（method_doc）：**

```yaml
gather:
  description: |
    The following figure shows the calculation process of Gather:

    .. image:: ../../images/Gather.png
```

**正确示例（op_def/yaml/doc）：**

```yaml
gather:
  description: |
    The following figure shows the common calculation process of Gather:

    .. image:: ../images/Gather.png
```

**注意：** 英文注释写法与中文文档保持一致。

**YAML格式示例汇总：**

```yaml
# function_doc示例
add:
  description: |
    add(input, other, *, alpha=1) -> Tensor
    Scales the `other` value by `alpha` and adds it to `input`.

  Args:
    input (Union[Tensor, number.Number, bool]): The first input.

  Keyword Args:
    alpha (number.Number, optional): A scaling factor. Default ``1``.

  Returns:
    Tensor, with the same shape as the broadcasted shape.

  Raises:
    TypeError: If the type of `other` is not one of the following.

  Supported Platforms:
    ``Ascend`` ``GPU`` ``CPU``

  Examples:
    >>> import numpy as np
    >>> import mindspore
    >>> from mindspore import Tensor, mint
    >>> x = Tensor(1, mindspore.int32)
    >>> y = Tensor(np.array([4, 5, 6]).astype(np.float32))
    >>> output = mint.add(x, y)
    >>> print(output)
    [5. 6. 7.]
```

```yaml
# method_doc示例（含重载）
add:
  description: |
    add(other) -> Tensor
    Adds other value to `self` element-wise.

  Args:
    other (Union[Tensor, number.Number, bool]): The other input.

  Returns:
    Tensor, with the same shape as the broadcasted shape.

  Supported Platforms:
    ``Ascend`` ``GPU`` ``CPU``

  Examples:
    >>> import numpy as np
    >>> import mindspore
    >>> from mindspore import Tensor
    >>> x = Tensor(np.array([1, 2, 3]))
    >>> y = Tensor(np.array([4, 5, 6]))
    >>> output = Tensor.add(x, y)
    >>> print(output)
    [5. 7. 9.]

  .. method:: Tensor.add(other, *, alpha=1) -> Tensor
      :noindex:

  Adds scaled other value to `self`.

  Keyword Args:
    alpha (number.Number): A scaling factor, default 1.

  Returns:
    Tensor, with the same shape as the broadcasted shape.

  Supported Platforms:
    ``Ascend`` ``GPU`` ``CPU``

  Examples:
    >>> x = Tensor(1, mindspore.int32)
    >>> y = Tensor(np.array([4, 5, 6]).astype(np.float32))
    >>> output = Tensor.add(x, y, alpha=0.5)
    >>> print(output)
    [3. 3.5 4.]
```

```yaml
# op_def/yaml/doc示例
gather:
  description: |
    Returns the slice of the input tensor corresponding to the elements of `input_indices`.

    .. image:: ../images/Gather.png

    .. note::
      - The value of input_indices must be in the range.

  Args:
    input_params (Tensor): The input Tensor.
    input_indices (Tensor): The specified indices.
    axis (int): The specified axis.

  Returns:
    Tensor

  Supported Platforms:
    ``Ascend`` ``GPU`` ``CPU``

  Examples:
    >>> import mindspore
    >>> input_params = mindspore.tensor([1, 2, 3, 4, 5, 6, 7])
    >>> input_indices = mindspore.tensor([0, 2, 4, 2, 6])
    >>> output = mindspore.ops.gather(input_params, input_indices, 0)
    >>> print(output)
    [1. 3. 5. 3. 7.]
```

**常见错误示例：**

```yaml
# 错误：description不是多行格式
add:
  description: Scales the `other` value.

# 错误：Args缩进不是4空格
add:
  Args:
  input (Tensor): description.

# 错误：Returns缺少类型
Returns:
  with the same shape

# 错误：Supported Platforms没有使用反引号
Supported Platforms:
  Ascend GPU CPU

# 错误：Examples缺少 >>> 代码前缀
Examples:
  import mindspore
  output = mindspore.add(x, y)

# 错误：重载缺少 :noindex:
  .. method:: Tensor.add(other, alpha=1) -> Tensor
```

---

## 常见错误汇总

### Python注释错误

1. Summary中包含冒号
2. Args参数类型与参数名之间缺少空格
3. Examples中 `>>>` 和 `...` 使用错误
4. 多行内容缺少空行
5. Note写成Notes
6. 链接格式错误
7. 引用其他API未使用 `:class:` 或 `:func:`
8. 变量名未使用反引号包裹
9. dtype格式错误

### YAML注释错误

1. description使用单行格式（应为 `description: |` 多行格式）
2. Args缩进不是4空格
3. 参数类型格式错误（如 `list[int]` 写成 `list:int`）
4. 可选参数格式错误（应为 `(类型, optional)`）
5. Returns缺少类型说明
6. Supported Platforms缺少反引号（应为 `` ``Ascend`` ``）
7. Examples缺少 `>>>` 代码前缀
8. 重载缺少 `:noindex:` 标签
9. 公式缩进错误（`.. math::` 内容需缩进）
10. 图片路径错误（method_doc应为 `../../images/`，op_def应为 `../images/`）

### 完整性错误

1. 缺失必需的注释项（Summary/Args/Returns/Raises）
2. 参数说明不完整
3. 返回值说明缺失
4. 示例代码不可运行
5. 异常说明缺失

---

## 格式验证要点

### 空格和缩进

- 参数名和类型之间有空格
- 不同类型内容间有空行
- `Args`/`Raises` 换行缩进4空格
- 二级参数与一级参数对齐

### 冒号使用

- 关键字后有冒号
- 参数名后有冒号
- Summary和Returns内容中无冒号

### 列表格式

- 同类型列表项间无空行
- 无序/有序列表内容与上方内容间有空行
- 二级列表正确缩进
