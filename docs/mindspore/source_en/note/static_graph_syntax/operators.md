# Static Graph Syntax —— Operators

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/static_graph_syntax/operators.md)

Arithmetic operators and assignment operators support the `Number` and `Tensor` operations, as well as the `Tensor` operations of different `dtype`.

## Unary Arithmetic Operators

| Unary Arithmetic Operator | Supported Type                               |
| :------------------------ | :------------------------------------------- |
| `+`                       | `Number`, `Tensor`, taking positive values.                           |
| `-`                       | `Number`, `Tensor`, `COOTensor`, `CSRTensor`, taking negative values. |
| `~`                       | `Tensor` with `Bool` data type, members take negation one by one.               |

notes:

- In native python the `~` operator get the bitwise inversion of its integer argument; in Mindspore the `~` redefined to get logic not for `Tensor(Bool)`.

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
| `&`     | `Number` & `Number`、`Tensor` & `Tensor`、`Number` & `Tensor`、`Tensor` & `Number`.                                                                                                                                                                  |
| `∣`      | `Number` &#124; `Number`、`Tensor` &#124; `Tensor`、`Number` &#124; `Tensor`、`Tensor` &#124; `Number`.                                                                                                                                                             |
| `^`     | `Number` ^ `Number`、`Tensor` ^ `Tensor`、`Number` ^ `Tensor`、`Tensor` ^ `Number`.                                                                                                                                                                  |
| `<<`    | `Number` << `Number`.                                                                                                                                                                                                                             |
| `>>`    | `Number` >> `Number`.                                                                                                                                                                                                                             |

Restrictions:

- If all operands are `Number` type, operations between  `Float64` and `Int32` are not supported. Operators including `+`, `-`, `*`, `/`, `%`, `**`, `//` all support left and right operands to be `Bool` value.
- If either operand is `Tensor` type, left and right operands can't both be `Bool` value.
- The `*` operation on `List/Tuple` and `Number` means that `List/Tuple` is copied from `Number` and then concatenated. The data type inside `List` can be any data type supported by the graph mode, and multi-layer nesting is also supported. The data type in `Tuple` must be `Number`, `String`, `None`, and multi-layer nesting is also supported.

## Assignment Operators

| Assignment Operator | Supported Type、 |
| :----------- |:--------|
| `=`          |All Built-in Python Types that MindSpore supported and MindSpore User-defined Data Types.|
| `+=` |`Number` += `Number`, `String` += `String`, `Number` += `Tensor`, `Tensor` += `Number`, `Tuple` += `Tensor`, `Tensor` += `Tuple`, `List` += `Tensor`, `Tensor` += `List`, `List` += `List`, `Tensor` += `Tensor`, `Tuple` += `Tuple`.|
| `-=` |`Number` -= `Number`, `Tensor` -= `Tensor`, `Number` -= `Tensor`, `Tensor` -= `Number`, `Tuple` -= `Tensor`, `Tensor` -= `Tuple`, `List` -= `Tensor`, `Tensor` -= `List`.|
| `*=` |`Number` \*= `Number`, `Tensor` \*= `Tensor`, `Number` \*= `Tensor`, `Tensor` \*= `Number`, `List` \*= `Number`, `Number` \*= `List`, `Tuple` \*= `Number`, `Number` \*= `Tuple`, `Tuple` \*= `Tensor`, `Tensor` \*= `Tuple`,  `List` \*= `Tensor`, `Tensor` \*= `List`.|
| `/=` |`Number` /= `Number`, `Tensor` /= `Tensor`, `Number` /= `Tensor`, `Tensor` /= `Number`, `Tuple` /= `Tensor`, `Tensor` /= `Tuple`, `List` /= `Tensor`, `Tensor` /= `List`.|
| `%=` |`Number` %= `Number`, `Tensor` %= `Tensor`, `Number` %= `Tensor`, `Tensor` %= `Number`, `Tuple` %= `Tensor`, `Tensor` %= `Tuple`,  `List` %= `Tensor`、`Tensor` %= `List`.|
| `**=` |`Number` \*\*= `Number`, `Tensor` \*\*= `Tensor`, `Number` \*\*= `Tensor`, `Tensor` \*\*= `Number`, `Tuple` \*\*= `Tensor`, `Tensor` \*\*= `Tuple`,  `List` \*\*= `Tensor`, `Tensor` \*\*= `List`.|
| `//=` |`Number` //= `Number`, `Tensor` //= `Tensor`, `Number` //= `Tensor`, `Tensor` //= `Number`, `Tuple` //= `Tensor`, `Tensor` //= `Tuple`, `List` //= `Tensor`, `Tensor` //= `List`.|
| `&=`     | `Number` &= `Number`、`Tensor` &= `Tensor`、`Number` &= `Tensor`、`Tensor` &= `Number`.                                                                                                                                                                              |
| `∣=`      | `Number` &#124;= `Number`、`Tensor` &#124;= `Tensor`、`Number` &#124;= `Tensor`、`Tensor` &#124;= `Number`.                                                                                                                                                         |
| `^=`     | `Number` ^= `Number`、`Tensor` ^= `Tensor`、`Number` ^= `Tensor`、`Tensor` ^= `Number`.                                                                                                                                                                              |
| `<<=`    | `Number` <<= `Number`.                                                                                                                                                                                                                                         |
| `>>=`    | `Number` >>= `Number`.                                                                                                                                                                                                                                         |

Constraints:

- If all operands of  `AugAssign` are `Number` type, value of Number can't be `Bool`.

- If all operands of  `AugAssign` are `Number` type, operations between  `Float64` and `Int32` are not supported.

- If either operand of  `AugAssign` is `Tensor` type, left and right operands can't both be `Bool` value.

- The `*=` operation on `List/Tuple` and `Number` means that `List/Tuple` is copied from `Number` and then concatenated, and the elements of the object within `List/Tuple` can contain any of the types supported by the intentional pattern, and multiple levels of nesting are also supported.

## Logical Operators

| Logical Operator | Supported Type|
| :----------- |:--------|
| `and` |`String`,  `Number`,  `Tuple`, `List` , `Dict`, `None`, `Scalar`, `Tensor`.|
| `or` |`String`,  `Number`,  `Tuple`, `List` , `Dict`, `None`, `Scalar`, `Tensor`.|
| `not` |`Number`, `tuple`, `List` and `Tensor`  with only one element.|

Restrictions:

- The left operand of operator `and`, `or` must be able to be converted to boolean value. For example, left operand can not be Tensor with multiple elements. If the left operand of `and`, `or` is variable `Tensor`, the right operand must also be single-element `Tensor` with the same type. Otherwise, there is no requirement for right operand.

## Compare Operators

| Compare Operator | Supported Type|
| :----------- |:--------|
| `in` |`Number` in `Tuple`, `String` in `Tuple`, `Tensor` in `Tuple`, `Number` in `List`, `String` in `List`, `Tensor` in `List`, and `String` in `Dictionary`.|
| `not in` | Same as `in`. |
| `is` | The value can only be `None`, `True`, or `False`. |
| `is not` | The value can only be `None`, `True`, or `False`. |
| < | `Number` < `Number`, `Number` < `Tensor`, `Tensor` < `Tensor`, `Tensor` < `Number`. |
| <= | `Number` <= `Number`, `Number` <= `Tensor`, `Tensor` <= `Tensor`, `Tensor` <= `Number`. |
| > | `Number` > `Number`, `Number` > `Tensor`, `Tensor` > `Tensor`, `Tensor` > `Number`. |
| >= | `Number` >= `Number`, `Number` >= `Tensor`, `Tensor` >= `Tensor`, `Tensor` >= `Number`. |
| != | `Number` != `Number` , `Number` != `Tensor`, `Tensor` != `Tensor`, `Tensor` != `Number`, `mstype` != `mstype`, `String` != `String`, `Tuple !` = `Tuple`, `List` != `List`. |
| == | `Number` == `Number`, `Number` == `Tensor`, `Tensor` == `Tensor`, `Tensor` == `Number`, `mstype` == `mstype`, `String` == `String`, `Tuple` == `Tuple`, `List` == `List`. |

Restrictions:

- For operators `<`, `<=`, `>`, `>=`, `!=`, if all operators are `Number` type, value of Number can't be `Bool`.
- For operators `<`, `<=`, `>`, `>=`, `!=`, `==`, if all operands are `Number` type, operations between  `Float64` and `Int32` are not supported.
- For operators `<`, `<=`, `>`, `>=`, `!=`, `==`, if either operand is `Tensor` type, left and right operands can't both be `Bool` value.
- For operator `==`, if all operands are `Number` type,  support both `Number` have `Bool` value, not support only one `Number` has `Bool` value.
- For operators `!=`, `==`, all supported types but `mstype` can compare with `None`.
- The chain comparison like: `a>b>c` is not supported.
