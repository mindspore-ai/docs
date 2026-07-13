# Python Docstring Guide (Python注释编写指南)

## Opening Description（起始描述）

- One-line summary in third person (e.g., "Returns", "Computes", "Checks")
- Include purpose/usage (e.g., "Usually used to extract a finite
  signal segment for FFT")
- Include math formula with `.. math::` if the operation involves
  mathematical computation
- Add variable definitions for math formulas (e.g., "where :math:`N`
  is the full window size")
- Description text must NOT contain colons (Arg format `name (type):` is exempt)

## Args（参数）

- `name (type): Description.` format
- For optional args: `name (type, optional): Description. Default: ``None``.`

## Keyword Args（关键字参数）

- Place after `Args:` when parameters follow a `*` separator
- Use `Keyword Args:` heading
- Same format as Args: `name (type): Description.`

## Returns（返回值）

- Full description including shape relation to inputs
- For elementwise ops: `Tensor, has the same shape and dtype as input.`
- For multiple returns: use numbered list or separate paragraphs
- Return value description must NOT contain colons — use commas instead

## Raises（异常）

- `name (Type): Description.` format (e.g., `ValueError: If `reduction` is not one of ``'none'``, ``'mean'``, ``'sum'``.`)
- Document all explicitly raised exceptions (except TypeError)
- Merge same-type exceptions for readability

## Examples（样例代码）

- Required in Python docstring
- Always runnable (copy-paste should work)
- Show imports, input creation, function call, and `print(output)`
  with expected output
- Use `>>>` for Python statements
- Heading must be `Examples:` (not `Example:`)

## Inputs / Outputs（输入/输出）

- Used for classes and Cell-like constructs instead of Args/Returns
- Inputs: `- **x** - Description of shape :math:`(*, in\_features)`.`
- Outputs: Description of the output tensor

## Supported Platforms（支持平台）

- May optionally document supported device types

---

## Reference (参考)

### Format Examples (格式示例)

#### Function Docstring

```python
def exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, is_stair=False):
    r"""
    Calculates learning rate based on exponential decay function. The learning rate for each step will
    be stored in a list.

    For the i-th step, the formula of computing decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = learning\_rate * decay\_rate^{\frac{current\_epoch}{decay\_epoch}}

    Where :math:`current\_epoch=floor(\frac{i}{step\_per\_epoch})`.

    Args:
        learning_rate (float): The initial value of learning rate.
        decay_rate (float): The decay rate.
        total_step (int): The total number of steps.
        step_per_epoch (int): The number of steps per epoch.
        decay_epoch (int): Number of epochs to decay over.
        is_stair (bool, optional): If true, learning rate decays once every `decay_epoch` times. Default: ``False`` .

    Returns:
        list[float]. The size of list is `total_step`.

    Raises:
        ValueError: If `learning_rate` or `decay_rate` is less than or equal to 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>>
        >>> learning_rate = 0.1
        >>> decay_rate = 0.9
        >>> total_step = 6
        >>> step_per_epoch = 2
        >>> decay_epoch = 1
        >>> lr = nn.exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch)
        >>> net = nn.Dense(2, 3)
        >>> optim = nn.SGD(net.trainable_params(), learning_rate=lr)
    """
```

#### Class Docstring

```python
class L1Loss(LossBase):
    r"""
    L1Loss is used to calculate the mean absolute error between the predicted value and the target value.

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad \text{with } l_n = \left| x_n - y_n \right|,

    where :math:`N` is the batch size. If `reduction` is not ``'none'``, then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    Args:
        reduction (str, optional): Apply a specific reduction method to the output:
            ``'none'``, ``'mean'``, ``'sum'``. Default: ``'mean'``.

    Inputs:
        - **logits** (Tensor) - Predicted value, Tensor of any dimension.
        - **labels** (Tensor) - Target value, same shape as the `logits` in common cases.

    Outputs:
        Tensor, data type is float.

    Raises:
        ValueError: If `reduction` is not one of ``'none'``, ``'mean'``, ``'sum'``.
        ValueError: If `logits` and `labels` have different shapes and cannot be broadcasted.

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> loss = nn.L1Loss()
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        0.33333334
    """
```

#### Method Docstring

```python
class Cell:
    ...

    def to(self, device=None, dtype=None, non_blocking=False):
        r"""
        Move and/or cast the parameters and buffers of this Cell (and all subcells).

        Args:
            device (str, optional): Target device, e.g. ``"Ascend"``, ``"CPU"``
                or ``"meta"``. Default: ``None``.
            dtype (mindspore.dtype, optional): Target dtype. Must be floating
                point or complex. Default: ``None``.
            non_blocking (bool, optional): If ``True``, the copy is performed
                asynchronously where possible. Default: ``False``.

        Returns:
            Cell, the cell itself.

        Raises:
            RuntimeError: Source tensor is on meta device and target is real.

        Note:
            - External references to ``Parameter`` objects transparently see the new data/device.

        Examples:
            >>> import mindspore as ms
            >>> import mindspore.nn as nn
            >>> net = nn.Dense(4, 4)
            >>> net.to(device="CPU", dtype=ms.float16)
        """
```

Note: For classes under ``mindspore.nn`` module, use **Inputs/Outputs** instead of **Args/Returns**
for methods' parameters.

### Conventions (规范)

#### String Format

- Use raw string `r"""..."""` to preserve backslashes in math notation
- First line should be a concise summary, then a blank line, then details

#### Section Headings

- `Args:`, `Returns:`, `Raises:`, `Keyword Args:`, `Inputs:`,
  `Outputs:`, `Examples:`, `Note:`

#### Examples Format

- Use `>>>` prefix for Python statements, expected output on next line without prefix
- Do NOT use `.. code-block:: python` inside docstrings
- Always runnable (copy-paste should work)

#### Special Directives

- `Note:` — plain section heading (singular, not `Notes:`), same level as `Args:`
- `.. warning::` — RST directive for deprecations or cautions. Content indented 4 spaces.
- `.. code-block::` — for non-Python snippets (shell, JSON, etc.). Blank line after directive, content indented 4 spaces.

```text
Note:
    This is an important note for the user.

.. warning::
    This API is deprecated. Use the new API instead.

.. code-block:: bash

    pip install mindspore
```

- Do NOT use `.. code-block:: python` for Python examples inside docstrings — use `>>>` instead

#### Cross-references

| Reference | Format |
| ----------- | -------- |
| Class | `` :class:`mindspore.nn.Cell` `` |
| Function | `` :func:`mindspore.ops.rsqrt` `` |
| Special method | `` :meth:`__getitem__` `` (from :class:`mindspore.dataset.dataloader.Dataset`) |
| External link | `` `link text <https://xxx>`_ `` |

#### Math Notation

```python
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

```python
# Float ops:
# Supported dtypes: float16, float32, float64, bfloat16.
# Integer ops:
# Supported dtypes: int8, int16, int32, int64, uint8.
# All numeric:
# Supported dtypes: int8, int16, int32, int64, uint8, uint16, uint32, uint64,
#     float16, float32, float64, bfloat16.
# Complex:
# Supported dtypes: complex64, complex128.
```

#### Shape Documentation

```python
# - input: Shape :math:`(N, C, H, W)`.
# - x: Shape :math:`(*)` where :math:`*` means any number of dimensions.
# - y: Shape :math:`(*)`, must be broadcastable with `x`.
```
