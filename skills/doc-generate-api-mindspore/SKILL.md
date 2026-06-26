---
name: doc-generate-api-mindspore
description: Generate documentation for Python APIs, functions, classes, and modules — including docstrings, API references, and examples. Use when adding docstrings to new functions or classes, writing API reference docs, creating examples, documenting classes, or following Python doc conventions. Triggers on phrases like "API documentation", "API docs", "document API", "write API documentation", "generate API docs", "API reference".
---

# API Documentation Guide

This skill generates professional Python API documentation following common Python conventions (NumPy-style, Google-style, or Sphinx-style).

## Before Generation

1. **Read the source code**: Understand the actual parameter types by reading the function implementation
2. **Verify parameter types**: Check if parameters accept `int`, `Tensor`, or other types
3. **Run examples**: Execute the example code to get the actual output values

## Output Target

The generated documentation should be directly added to the Python source file (.py) as the function/class docstring. Write the docstring into the actual source file using the Edit tool.

## Docstring Format

### Function Docstring

```python
def rsqrt(input):
    r"""
    Returns reciprocal of the square root of a tensor element-wise.

    .. math::

        out_{i} = \frac{1}{\sqrt{input_{i}}}

    Args:
        input (Tensor): The input tensor. Supported dtypes: float16, float32, float64,
            bfloat16. Shape: :math:`(*)` where :math:`*` means any number of dimensions.

    Returns:
        Tensor, has the same shape and dtype as `input`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If dtype of `input` is not float16, float32, float64 or bfloat16.

    Examples:
        >>> import numpy as np
        >>> input = np.array([0.25, 4.0, 1.0])
        >>> output = rsqrt(input)
        >>> print(output)
        [2.  0.5 1. ]
    """
```

### Class Docstring

```python
class Linear:
    r"""
    Applies a linear transformation to the input: :math:`y = xA^T + b`.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        has_bias (bool): If set to False, the layer will not learn an additive bias.
            Default: ``True``.
        dtype: Data type of the weight and bias. Default: ``float32``.

    Inputs:
        - **x** - Input of shape :math:`(*, in\_features)`.

    Outputs:
        Tensor of shape :math:`(*, out\_features)`.

    Raises:
        TypeError: If `in_features` or `out_features` is not an int.

    Examples:
        >>> import numpy as np
        >>> x = np.array([[1.0, 2.0, 3.0]])
        >>> net = Linear(3, 4)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 4)
    """
```

Note: For classes, use **Inputs/Outputs** instead of **Args/Returns** for methods' parameters.

## Section-by-Section Guide

### Opening description

- One-line summary of what the function/class does
- Use "element-wise" for elementwise operations
- **Include purpose/usage**: Add a sentence explaining what the function is typically used for (e.g., "Usually used to extract a finite signal segment for FFT", "A triangular-shaped weighting function used for smoothing or frequency analysis of signals in digital signal processing")
- Include math formula with `.. math::` if the operation involves mathematical computation
- **Add variable definitions**: For math formulas, explain the variables (e.g., "where :math:`N` is the full window size, and n is a natural number less than :math:`N` :[0, 1, ..., N-1]")

### Args

- `name (type): Description.` format
- List supported dtypes explicitly if applicable
- Describe shape with ``:math:`(*, H, W)``` for fixed dims, ``:math:`(*)``` for arbitrary
- For optional args: `name (type, optional): Description. Default: ``None``.`

### Returns

- Full description including shape relation to inputs
- `Tensor, has the same shape and dtype as input.` for elementwise operations
- For multiple returns: use numbered list or separate paragraphs

### Raises

- `TypeError`: for wrong input type
- `ValueError`: for wrong value/shape
- Document all explicitly raised exceptions

### Examples

- Always runnable (copy-paste should work)
- Show imports, input creation, the function call, and `print(output)` with expected output
- Use `>>>` prompt for Python statements, show expected output below

## Math Notation

Use reStructuredText math directives:

```rst
Inline: :math:`x^2 + y^2`

Block:
.. math::

    \text{out}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
```

**Piecewise function** (for functions withDifferent definitions in Different ranges):

```rst
.. math::

    w[n] = 1 - \left| \frac{2n}{N-1} - 1 \right| = \begin{cases}
    \frac{2n}{N - 1} & \text{if } 0 \leq n \leq \frac{N - 1}{2} \\
    2 - \frac{2n}{N - 1} & \text{if } \frac{N - 1}{2} < n < N \\
    \end{cases},

where :math:`N` is the full window size, and n is a natural number less than :math:`N` :[0, 1, ..., N-1].
```

Common patterns:

- Element-wise: `out_{i} = f(input_{i})`
- Matrix: `Y = xA^T + b`
- Reduction: `out = \sum_{i} input_{i}`

## dtype Documentation Patterns

```python
# Common dtype lists for the docstring:

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

## Shape Documentation

```python
# For ops with specific shape requirements:
# - input: Shape :math:`(N, C, H, W)`.
# - kernel_size (int or tuple[int]): The size of the sliding window.
#   If int, the height and width will be the same value.

# For broadcast-compatible inputs:
# - x: Shape :math:`(*)` where :math:`*` means any number of dimensions.
# - y: Shape :math:`(*)`, must be broadcastable with `x`.
```

## Common Docstring Styles

### NumPy Style

```python
def func(x):
    """
    One-line summary.

    Longer description.

    Parameters
    ----------
    x : type
        Description.

    Returns
    -------
    result : type
        Description.
    """
```

### Google Style

```python
def func(x):
    """
    One-line summary.

    Longer description.

    Args:
        x: Description.

    Returns:
        Description.
    """
```

### Sphinx Style (reStructuredText)

```python
def func(x):
    """
    One-line summary.

    :param x: Description.
    :type x: type
    :return: Description.
    :rtype: type
    """
```

## Reference from Existing Docs

When generating documentation for a new API, you can reference existing well-documented APIs in the codebase to ensure consistency:

1. **Find similar APIs**: Search for functions/classes with similar functionality in the codebase
2. **Extract patterns**: Note the docstring structure, section ordering, and style
3. **Adapt format**: Apply the same patterns to your new documentation
4. **Verify consistency**: Ensure parameter names, types, and return descriptions match the reference

Example workflow:

```bash
# Find similar functions in the codebase
grep -r "def similar_function" --include="*.py" .

# Find well-documented classes
grep -r "class.*:" -A 30 *.py | head -100
```

## Quality Checklist

- [ ] Docstring present on the Python function/class
- [ ] Opening description includes purpose/usage explanation
- [ ] Math formula included for non-obvious operations
- [ ] Variable definitions added for math formulas (where N, n, M, etc. are explained)
- [ ] All parameters documented with type and description
- [ ] Returns section accurately describes output
- [ ] Raises section lists all explicitly raised exceptions
- [ ] Example is copy-paste runnable and shows expected output
- [ ] Parameters match the actual function signature
- [ ] Type hints are consistent with parameter descriptions
- [ ] Default values are documented for optional parameters