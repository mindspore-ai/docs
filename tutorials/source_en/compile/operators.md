# Graph Mode Syntax - Operators

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.9.0/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/r2.9.0/tutorials/source_en/compile/operators.md)

Arithmetic operators and assignment operators support the `Number` and `Tensor` operations, as well as the `Tensor` operations of different `dtype`.

## Unary Arithmetic Operators

| Unary Arithmetic Operator | Supported Type                               |
| :------------------------ | :------------------------------------------- |
| `+`                       | `Number`, `Tensor`, taking positive values.                           |
| `-`                       | `Number`, `Tensor`, `COOTensor`, `CSRTensor`, taking negative values. |
| `~`                       | `Tensor` with `bool` data type, members take negation one by one.               |

Notes:

- In native Python, the `~` operator gets the bitwise inversion of its integer argument; in MindSpore, `~` is redefined to get logical NOT for `Tensor(Bool)`.

## Binary Arithmetic Operators

| Binary Arithmetic Operator | Supported Type|
| :----------- |:--------|
| `+` |`Number` + `Number`, `String` + `String`, `Number` + `Tensor`, `Tensor` + `Number`, `Tuple` + `Tensor`, `Tensor` + `Tuple`, `List` + `Tensor`, `Tensor`+`List`, `List`+`List`, `Tensor` + `Tensor`, `Tuple` + `Tuple`, `COOTensor` + `Tensor`, `Tensor` + `COOTensor`, `COOTensor` + `COOTensor`, `CSRTensor` + `CSRTensor`.|
| `-` |`Number` - `Number`, `Tensor` - `Tensor`, `Number` -`Tensor`, `Tensor` - `Number`, `Tuple` -`Tensor`, `Tensor` -`Tuple`, `List` -`Tensor`, `Tensor` -`List`, `COOTensor` - `Tensor`, `Tensor` - `COOTensor`, `COOTensor` - `COOTensor`, `CSRTensor` - `CSRTensor`.|
| `*` |`Number` \* `Number`, `Tensor` \* `Tensor`, `Number` \* `Tensor`, `Tensor` \* `Number`, `List` \* `Number`, `Number` \* `List`, `Tuple` \* `Number`, `Number` \* `Tuple`, `Tuple` \* `Tensor`, `Tensor` \* `Tuple`,  `List` \*`Tensor`, `Tensor` \* `List`, `COOTensor` \* `Tensor`, `Tensor` \* `COOTensor`, `CSRTensor` \* `Tensor`, `Tensor` \* `CSRTensor`.|
| `/` |`Number` / `Number`, `Tensor` / `Tensor`, `Number` / `Tensor`, `Tensor` / `Number`, `Tuple` / `Tensor`, `Tensor` / `Tuple`,  `List` / `Tensor`, `Tensor` / `List`, `COOTensor` / `Tensor`, `CSRTensor` / `Tensor`.|
| `%` |`Number` % `Number`, `Tensor` % `Tensor`, `Number` % `Tensor`, `Tensor` % `Number`, `Tuple` % `Tensor`, `Tensor` % `Tuple`, `List` % `Tensor`, `Tensor` % `List`.|
| `**` |`Number` \*\* `Number`, `Tensor` \*\* `Tensor`, `Number` \*\* `Tensor`, `Tensor` \*\* `Number`, `Tuple` \*\* `Tensor`, `Tensor` \*\* `Tuple`,  `List` \*\* `Tensor`, `Tensor` \*\* `List`.|
| `//` |`Number` // `Number`, `Tensor` // `Tensor`, `Number` // `Tensor`, `Tensor` // `Number`, `Tuple` // `Tensor`, `Tensor` // `Tuple`,  `List` // `Tensor`, `Tensor` // `List`.|
| `&`     | `Number` & `Number`, `Tensor` & `Tensor`, `Number` & `Tensor`, `Tensor` & `Number`.                                                                                                                                                                  |
| `∣`      | `Number` &#124; `Number`, `Tensor` &#124; `Tensor`, `Number` &#124; `Tensor`, `Tensor` &#124; `Number`.                                                                                                                                                             |
| `^`     | `Number` ^ `Number`, `Tensor` ^ `Tensor`, `Number` ^ `Tensor`, `Tensor` ^ `Number`.                                                                                                                                                                  |
| `<<`    | `Number` << `Number`.                                                                                                                                                                                                                             |
| `>>`    | `Number` >> `Number`.                                                                                                                                                                                                                             |
| `@`    | `Tensor` @ `Tensor`.                                                                                                                                                                                                                             |

Restrictions:

- If all operands are `Number` type, operations between `float64` and `int32` are not supported. Operators including `+`, `-`, `*`, `/`, `%`, `**`, `//` all support left and right operands to be `bool` values.
- If either operand is `Tensor` type, left and right operands can't both be `bool` value.
- The `*` operation on `List/Tuple` and `Number` means that `List/Tuple` is copied from `Number` and then concatenated. The data type inside `List` can be any data type supported by the graph mode, and multi-layer nesting is also supported. The data type in `Tuple` must be `Number`, `String`, `None`, and multi-layer nesting is also supported.

## Assignment Operators

| Assignment Operator | Supported Type |
| :----------- |:--------|
| `=`          |All Built-in Python Types that MindSpore supported and MindSpore User-defined Data Types.|
| `+=` |`Number` += `Number`, `String` += `String`, `Number` += `Tensor`, `Tensor` += `Number`, `Tuple` += `Tensor`, `Tensor` += `Tuple`, `List` += `Tensor`, `Tensor` += `List`, `List` += `List`, `Tensor` += `Tensor`, `Tuple` += `Tuple`.|
| `-=` |`Number` -= `Number`, `Tensor` -= `Tensor`, `Number` -= `Tensor`, `Tensor` -= `Number`, `Tuple` -= `Tensor`, `Tensor` -= `Tuple`, `List` -= `Tensor`, `Tensor` -= `List`.|
| `*=` |`Number` \*= `Number`, `Tensor` \*= `Tensor`, `Number` \*= `Tensor`, `Tensor` \*= `Number`, `List` \*= `Number`, `Number` \*= `List`, `Tuple` \*= `Number`, `Number` \*= `Tuple`, `Tuple` \*= `Tensor`, `Tensor` \*= `Tuple`,  `List` \*= `Tensor`, `Tensor` \*= `List`.|
| `/=` |`Number` /= `Number`, `Tensor` /= `Tensor`, `Number` /= `Tensor`, `Tensor` /= `Number`, `Tuple` /= `Tensor`, `Tensor` /= `Tuple`, `List` /= `Tensor`, `Tensor` /= `List`.|
| `%=` |`Number` %= `Number`, `Tensor` %= `Tensor`, `Number` %= `Tensor`, `Tensor` %= `Number`, `Tuple` %= `Tensor`, `Tensor` %= `Tuple`,  `List` %= `Tensor`, `Tensor` %= `List`.|
| `**=` |`Number` \*\*= `Number`, `Tensor` \*\*= `Tensor`, `Number` \*\*= `Tensor`, `Tensor` \*\*= `Number`, `Tuple` \*\*= `Tensor`, `Tensor` \*\*= `Tuple`,  `List` \*\*= `Tensor`, `Tensor` \*\*= `List`.|
| `//=` |`Number` //= `Number`, `Tensor` //= `Tensor`, `Number` //= `Tensor`, `Tensor` //= `Number`, `Tuple` //= `Tensor`, `Tensor` //= `Tuple`, `List` //= `Tensor`, `Tensor` //= `List`.|
| `&=`     | `Number` &= `Number`, `Tensor` &= `Tensor`, `Number` &= `Tensor`, `Tensor` &= `Number`.                                                                                                                                                                              |
| `∣=`      | `Number` &#124;= `Number`, `Tensor` &#124;= `Tensor`, `Number` &#124;= `Tensor`, `Tensor` &#124;= `Number`.                                                                                                                                                         |
| `^=`     | `Number` ^= `Number`, `Tensor` ^= `Tensor`, `Number` ^= `Tensor`, `Tensor` ^= `Number`.                                                                                                                                                                              |
| `<<=`    | `Number` <<= `Number`.                                                                                                                                                                                                                                         |
| `>>=`    | `Number` >>= `Number`.                                                                                                                                                                                                                                         |
| `@=`    | `Tensor` @= `Tensor`.                                                                                                                                                                                                                                         |

Constraints:

- If all operands of `AugAssign` are `Number` type, the value of Number can't be `bool`.

- If all operands of `AugAssign` are `Number` type, operations between `float64` and `int32` are not supported.

- If either operand of `AugAssign` is `Tensor` type, left and right operands can't both be `bool` value.

- The `*=` operation on `List/Tuple` and `Number` means that `List/Tuple` is copied from `Number` and then concatenated, and the elements of the object within `List/Tuple` can contain any of the types supported by the graph mode, and multiple levels of nesting are also supported.

## Logical Operators

| Logical Operator | Supported Type|
| :----------- |:--------|
| `and` |`String`,  `Number`,  `Tuple`, `List` , `Dict`, `None`, `Scalar`, `Tensor`.|
| `or` |`String`,  `Number`,  `Tuple`, `List` , `Dict`, `None`, `Scalar`, `Tensor`.|
| `not` |`Number`, `Tuple`, `List` and `Tensor` with only one element.|

Restrictions:

- The left operand of operator `and` or `or` must be able to be converted to a boolean value. For example, the left operand cannot be a Tensor with multiple elements. If the left operand of `and` or `or` is a variable `Tensor`, the right operand must also be a single-element `Tensor` with the same type. Otherwise, there is no requirement for the right operand.

## Compare Operators

| Compare Operator | Supported Type|
| :----------- |:--------|
| `in` |`Number` in `Tuple`, `String` in `Tuple`, `Tensor` in `Tuple`, `Number` in `List`, `String` in `List`, `Tensor` in `List`, `String` in `Dictionary`, `Number` in `Dictionary`, constant `Tensor` in `Dictionary`, and `Tuple` in `Dictionary`.|
| `not in` | Same as `in`. |
| `is` | The value can only be `None`, `True`, or `False`. |
| `is not` | The value can only be `None`, `True`, or `False`. |
| < | `Number` < `Number`, `Number` < `Tensor`, `Tensor` < `Tensor`, `Tensor` < `Number`. |
| <= | `Number` <= `Number`, `Number` <= `Tensor`, `Tensor` <= `Tensor`, `Tensor` <= `Number`. |
| > | `Number` > `Number`, `Number` > `Tensor`, `Tensor` > `Tensor`, `Tensor` > `Number`. |
| >= | `Number` >= `Number`, `Number` >= `Tensor`, `Tensor` >= `Tensor`, `Tensor` >= `Number`. |
| != | `Number` != `Number`, `Number` != `Tensor`, `Tensor` != `Tensor`, `Tensor` != `Number`, `mstype` != `mstype`, `String` != `String`, `Tuple` != `Tuple`, `List` != `List`. |
| == | `Number` == `Number`, `Number` == `Tensor`, `Tensor` == `Tensor`, `Tensor` == `Number`, `mstype` == `mstype`, `String` == `String`, `Tuple` == `Tuple`, `List` == `List`. |

Restrictions:

- For operators `<`, `<=`, `>`, `>=`, `!=`, if all operands are of `Number` type, the value of Number can't be `bool`.
- For operators `<`, `<=`, `>`, `>=`, `!=`, `==`, if all operands are of `Number` type, operations between `float64` and `int32` are not supported.
- For operators `<`, `<=`, `>`, `>=`, `!=`, `==`, if either operand is of `Tensor` type, left and right operands can't both be `bool` value.
- For operator `==`, if all operands are of `Number` type, both `Number` operands can have `bool` values, but having only one `Number` with a `bool` value is not supported.
- For operators `!=`, `==`, all supported types except `mstype` can be compared with `None`.
