# Side Effects

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/design/side_effect.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Concepts

### Pure Function

A function whose return value depends only on the real parameters of the function and has no side effects is a pure function.
A pure function is closer to a function in the mathematical sense: for the same input parameters, users always get the same return value.
If the program contains only pure functions, the order in which they are evaluated will not affect the program result.
For example, in the following code, assuming that `add` is a pure function, the order in which `a` and `b` are evaluated will not affect the result of `c`.

```python
    a = add(1, 2)
    b = add(3, 4)
    c = add(a, b)
```

### Side Effects

A function has side effects if it changes the external state.
Or there are other observable effects occurring besides the return value of the function in the functions with side effects.
For example modifying global variables, modifying the value of reference type parameters, executing input and output operations, calling other functions with side effects.
When there are side effects, the behavior of the program may change depending on the different order in which the values are evaluated.
For example, in the following code, suppose `add` is a pure function and `assign` is a function with side effects (it changes the input parameter x), the different order in which `a`, `b` and `c` are evaluated will lead to different results for `d`.

```python
    a = add(1, x)
    b = assign(x, 100)  # side effect
    c = add(3, x)
    d = add(a, c)
```

Because of the side effects, `a`, `b` and `c` in the above program should be evaluated strictly in the order in which they are in the code, otherwise they will produce unintended results.

## Design

MindSpore uses a functional intermediate representation based on a graph representation, and refer to [MindIR](https://www.mindspore.cn/docs/en/master/design/mindir.html).
Conceptually, the functions in MindIR are pure functions and do not have side effects.
However, MindSpore can support computational models with side effects and provide operators with side effects, such as optimizer operators that will directly modify the input parameters.
In order to support operator and computational models with side effects, MindSpore converts the side effects in the code to pure functional form when compiling the model. This ensures that computations with side effects are executed in the desired order while keeping MindIR pure functional semantics.

### Converting Side Effects to Pure Functions

To be able to convert a function with side effects to a pure function form, MindSpore treats the external state affected by the side effect function as a data object. The modification of the external state by the function is then converted to a state object as the input to the function, and the modified state object is returned:

```python
    ret = func_with_side_effect(args)
```

converted as:

```python
    ret, state1 = pure_func(args, state0)
```

Here the return value of `pure_func` depends only on the input parameters, and the input state `state0` is unchanged and the updated state `state1` is returned, so it can be seen as a pure function.

### Intermediate Representation of Side Effects

Since MindIR functions do not support multiple return values, MindSpore introduces a virtual operator `UpdateState`. The above `pure_func` function is expressed as an intermediate representation of the following form:

```python
    ret = pure_func(args, state0)
    state1 = UpdateState(state0, ret)
```

In addition, to ensure the correct order of reading and writing, MindSpore introduces a `Load` operator. If the input to a function is a global parameter, a `Load` is inserted to ensure that the function reads the correct parameter value.
For example, `add` in the following code needs to read in a global parameter `param`:

```python
    out = add(self.param, x)
```

MindSpore converts this to an intermediate representation of the following form:

```python
    p = Load(self.param, state0)
    state1 = UpdateState(state0, p)
    out = add(p, x)
```

### Classifications of Side Effects

MindSpore classifies side effects into three types, depending on the different external state types influenced by side effects:

1. Memory side effects: Affecting the state in memory, such as modifying global variables, modifying input parameters.

2. Input and output side effects: With input and output operations, such as printing information to the console.

3. Hidden side effects: There is no obvious external state change, but there is an actual hidden state change. For example, the random number generation operator affects the state of the random number generator.

In MindSpore, memory side effects and input and output side effects are represented by separate state objects, so these two types of side effects are represented as two separate execution sequences.

Hidden side effects are not reflected as separate state objects and execution sequences because there is no explicit external state counterpart, but MindSpore internally performs some special processing on it, such as preventing the fusion of two random number generation operators to prevent generating wrong results.

### Side Effect Operator Mark

Operators are marked whether there are side effects by adding specific attributes. MindSpore supports the following attributes to mark the side effects of an operator.

- side_effect_mem: Memory side effect
- side_effect_io: Input and output side effect
- side_effect_hidden: Hidden side effect

For example, to mark an operator as having memory side effects:

```python
    @prim_attr_register
    def __init__(self):
        ...
        self.add_prim_attr('side_effect_mem', True)
```

MindSpore can ensure that the side effects are executed in the desired order only if they are correctly identified.

## Related Scenarios

MindSpore automatically identifies side effects in the code and ensures that they are executed in the correct order.

Therefore, in the great majority of cases, model developers and users do not need to be concerned about whether the model has side effects and how to ensure the correct order of execution.

### Operator Development

If the developed operator is considered to have side effects, it needs to be correctly identified that there are side effects and what kind of side effects they are by the operator properties. Otherwise, there is a risk that the model by using the operator may lead to incorrect results because the evaluation sequence is not performed as expected.

### Model Development

Typically, model developers do not need to be concerned with side effects, but understanding the side effect rationale may be helpful in anticipating the order of code execution. Also by knowing which operators have side effects, one can make better operator choices.

### MindIR

If the model has side effects, `UpdateState` and `Load` nodes exist in the exported MindIR, and their role is to handle side effects and order preservation.
