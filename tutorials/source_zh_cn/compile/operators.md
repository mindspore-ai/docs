# 图模式语法-运算符

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_zh_cn/compile/operators.md)

算术运算符和赋值运算符支持`Number`和`Tensor`运算，也支持不同`dtype`的`Tensor`运算。

## 单目算术运算符

| 单目算术运算符 | 支持类型                                        |
| :------------- | :---------------------------------------------- |
| `+`            | `Number`、`Tensor`，取正值。                    |
| `-`            | `Number`、`Tensor`、`COOTensor`、`CSRTensor`，取负值。 |
| `~`            | `Tensor`，且其数据类型为`bool`。成员逐个取反。 |

说明：

- 在Python中`~`操作符对输入的整数按位取反; MindSpore对`~`的功能重新定义为对`Tensor(Bool)`的逻辑取反。

## 二元算术运算符

| 二元算术运算符 | 支持类型                                                     |
| :------------- | :----------------------------------------------------------- |
| `+`            | `Number` + `Number`、`String` + `String`、`Number` + `Tensor`、`Tensor` + `Number`、`Tuple` + `Tensor`、`Tensor` + `Tuple`、`List` + `Tensor`、`Tensor`+`List`、`List`+`List`、`Tensor` + `Tensor`、`Tuple` + `Tuple`、`COOTensor` + `Tensor`、`Tensor` + `COOTensor`、`COOTensor` + `COOTensor`、`CSRTensor` + `CSRTensor`。 |
| `-`            | `Number` - `Number`、`Tensor` - `Tensor`、`Number` - `Tensor`、`Tensor` - `Number`、`Tuple` - `Tensor`、`Tensor` - `Tuple`、`List` - `Tensor`、`Tensor` - `List`、`COOTensor` - `Tensor`、`Tensor` - `COOTensor`、`COOTensor` - `COOTensor`、`CSRTensor` - `CSRTensor`。 |
| `*`            | `Number` \* `Number`、`Tensor` \* `Tensor`、`Number` \* `Tensor`、`Tensor` \* `Number`、`List` \* `Number`、`Number` \* `List`、`Tuple` \* `Number`、`Number` \* `Tuple`、`Tuple` \* `Tensor`、`Tensor` \* `Tuple`、 `List` \* `Tensor`、`Tensor` \* `List`、`COOTensor` \* `Tensor`、`Tensor` \* `COOTensor`、`CSRTensor` \* `Tensor`、`Tensor` \* `CSRTensor`。 |
| `/`            | `Number` / `Number`、`Tensor` / `Tensor`、`Number` / `Tensor`、`Tensor` / `Number`、`Tuple` / `Tensor`、`Tensor` / `Tuple`、`List` / `Tensor`、`Tensor` / `List`、`COOTensor` / `Tensor`、`CSRTensor` / `Tensor`。 |
| `%`            | `Number` % `Number`、`Tensor` % `Tensor`、`Number` % `Tensor`、`Tensor` % `Number`、`Tuple` % `Tensor`、`Tensor` % `Tuple`、`List` % `Tensor`、`Tensor` % `List`。 |
| `**`           | `Number` \*\* `Number`、`Tensor` \*\* `Tensor`、`Number` \*\* `Tensor`、`Tensor` \*\* `Number`、`Tuple` \*\* `Tensor`、`Tensor` \*\* `Tuple`、 `List` \*\* `Tensor`、`Tensor` \*\* `List`。 |
| `//`           | `Number` // `Number`、`Tensor` // `Tensor`、`Number` // `Tensor`、`Tensor` // `Number`、`Tuple` // `Tensor`、`Tensor` // `Tuple`、`List` // `Tensor`、`Tensor` // `List`。 |
| `&`     | `Number` & `Number`、`Tensor` & `Tensor`、`Number` & `Tensor`、`Tensor` & `Number`。                                                                                                                                                                  |
| `∣`      | `Number` &#124; `Number`、`Tensor` &#124; `Tensor`、`Number` &#124; `Tensor`、`Tensor` &#124; `Number`。                                                                                                                                                             |
| `^`     | `Number` ^ `Number`、`Tensor` ^ `Tensor`、`Number` ^ `Tensor`、`Tensor` ^ `Number`。                                                                                                                                                                  |
| `<<`    | `Number` << `Number`。                                                                                                                                                                                                                             |
| `>>`    | `Number` >> `Number`。                                                                                                                                                                                                                             |
| `@`     | `Tensor` @ `Tensor`。                                                                                                                                                                                                                             |

限制：

- 当左右操作数都为`number`类型时，不支持`float64` 和 `int32`间的运算。`+`、`-`、`*`、`/`、`%`、`**`、`//` 支持左右操作数的值同时为`bool`。
- 当任一操作数为`tensor`类型时，左右操作数的值不可同时为`bool`。
- `list/tuple`和`number`进行`*`运算时表示将`list/tuple`复制`number`份后串联起来，`list`内的数据类型可以是图模式下支持的任意数据类型，也支持多层嵌套。`tuple`内的数据类型必须为`number`、`string`、`none`，也支持多层嵌套。

## 赋值运算符

| 赋值运算符 | 支持类型                                                     |
| :--------- | :----------------------------------------------------------- |
| `=`        | MindSpore支持的Python内置数据类型和MindSpore自定义数据类型   |
| `+=`       | `Number` += `Number`、`String` += `String`、`Number` += `Tensor`、`Tensor` += `Number`、`Tuple` += `Tensor`、`Tensor` += `Tuple`、`List` += `Tensor`、`Tensor` += `List`、`List` += `List`、`Tensor` += `Tensor`、`Tuple` += `Tuple`。 |
| `-=`       | `Number` -= `Number`、`Tensor` -= `Tensor`、`Number` -= `Tensor`、`Tensor` -= `Number`、`Tuple` -= `Tensor`、`Tensor` -= `Tuple`、`List` -= `Tensor`、`Tensor` -= `List`。 |
| `*=`       | `Number` \*= `Number`、`Tensor` \*= `Tensor`、`Number` \*= `Tensor`、`Tensor` \*= `Number`、`List` \*= `Number`、`Number` \*= `List`、`Tuple` \*= `Number`、`Number` \*= `Tuple`、`Tuple` \*= `Tensor`、`Tensor` \*= `Tuple`、 `List` \*= `Tensor`、`Tensor` \*= `List`。 |
| `/=`       | `Number` /= `Number`、`Tensor` /= `Tensor`、`Number` /= `Tensor`、`Tensor` /= `Number`、`Tuple` /= `Tensor`、`Tensor` /= `Tuple`、`List` /= `Tensor`、`Tensor` /= `List`。 |
| `%=`       | `Number` %= `Number`、`Tensor` %= `Tensor`、`Number` %= `Tensor`、`Tensor` %= `Number`、`Tuple` %= `Tensor`、`Tensor` %= `Tuple`、`List` %= `Tensor`、`Tensor` %= `List`。 |
| `**=`      | `Number` \*\*= `Number`、`Tensor` \*\*= `Tensor`、`Number` \*\*= `Tensor`、`Tensor` \*\*= `Number`、`Tuple` \*\*= `Tensor`、`Tensor` \*\*= `Tuple`、 `List` \*\*= `Tensor`、`Tensor` \*\*= `List`。 |
| `//=`      | `Number` //= `Number`、`Tensor` //= `Tensor`、`Number` //= `Tensor`、`Tensor` //= `Number`、`Tuple` //= `Tensor`、`Tensor` //= `Tuple`、`List` //= `Tensor`、`Tensor` //= `List`。 |
| `&=`     | `Number` &= `Number`、`Tensor` &= `Tensor`、`Number` &= `Tensor`、`Tensor` &= `Number`。                                                                                                                                                                              |
| `∣=`      | `Number` &#124;= `Number`、`Tensor` &#124;= `Tensor`、`Number` &#124;= `Tensor`、`Tensor` &#124;= `Number`。                                                                                                                                                         |
| `^=`     | `Number` ^= `Number`、`Tensor` ^= `Tensor`、`Number` ^= `Tensor`、`Tensor` ^= `Number`。                                                                                                                                                                              |
| `<<=`    | `Number` <<= `Number`。                                                                                                                                                                                                                                         |
| `>>=`    | `Number` >>= `Number`。                                                                                                                                                                                                                                         |
| `@=`     | `Tensor` @= `Tensor`。                                                                                                                                                                                                                                         |

限制：

- 当`AugAssign`的左右操作数都为`number`类型时，`number`的值不可为`bool` 类型。

- 当`AugAssign`的左右操作数都为`number`类型时，不支持`float64` 和 `int32`间的运算。

- 当`AugAssign`的任一操作数为`tensor`类型时，左右操作数的值不可同时为`bool`。

- `list/tuple`和`number`进行`*=`运算时表示将`list/tuple`复制`number`份后串联起来，`list/tuple`内对象的元素可以包含任意图模式支持的类型，也支持多层嵌套。

## 逻辑运算符

| 逻辑运算符 | 支持类型                                                     |
| :--------- | :----------------------------------------------------------- |
| `and`      | `String`、 `Number`、 `Tuple`、`List` 、`Dict`、`None`、标量、Tensor。 |
| `or`       | `String`、 `Number`、 `Tuple`、`List` 、`Dict`、`None`、标量、Tensor。 |
| `not`      | `Number`、`Tuple`、`List`、只有一个成员的Tensor。            |

限制：

- `and`、`or`的左操作数必须要能被转换成布尔值。例如：左操作数不能为存在多个元素的Tensor。当`and`、`or`的左操作数是变量Tensor时，右操作数必须也是同类型Tensor且Tensor成员个数只能有一个。在其余情况下，右操作数无要求。

## 比较运算符

| 比较运算符 | 支持类型                                                     |
| :--------- | :----------------------------------------------------------- |
| `in`       | `Number` in `Tuple`、`String` in `Tuple`、`Tensor` in `Tuple`、`Number` in `List`、`String` in `List`、`Tensor` in `List`、`String` in `Dictionary`、`Number` in `Dictionary`、常量`Tensor` in `Dictionary`、 `Tuple` in `Dictionary`。|
| `not in`   | 与`in`相同。                                                 |
| `is`       | 仅支持判断是`None`、 `True`或者`False`。                     |
| `is not`   | 仅支持判断不是`None`、 `True`或者`False`。                   |
| <          | `Number` < `Number`、`Number` < `Tensor`、`Tensor` < `Tensor`、`Tensor` < `Number`。 |
| <=         | `Number` <= `Number`、`Number` <= `Tensor`、`Tensor` <= `Tensor`、`Tensor` <= `Number`。 |
| >          | `Number` > `Number`、`Number` > `Tensor`、`Tensor` > `Tensor`、`Tensor` > `Number`。 |
| >=         | `Number` >= `Number`、`Number` >= `Tensor`、`Tensor` >= `Tensor`、`Tensor` >= `Number`。 |
| !=         | `Number` != `Number`、`Number` != `Tensor`、`Tensor` != `Tensor`、`Tensor` != `Number`、`mstype` != `mstype`、`String` != `String`、`Tuple !` = `Tuple`、`List` != `List`。 |
| ==         | `Number` == `Number`、`Number` == `Tensor`、`Tensor` == `Tensor`、`Tensor` == `Number`、`mstype` == `mstype`、`String` == `String`、`Tuple` == `Tuple`、`List` == `List`。 |

限制：

- 对于`<`、`<=`、`>`、`>=`、`!=`来说，当左右操作数都为`number`类型时，`number`的值不可为`bool` 类型。
- 对于`<`、`<=`、`>`、`>=`、`!=`、`==`来说，当左右操作数都为`number`类型时，不支持`float64` 和 `int32`间的运算。
- 对于`<`、`<=`、`>`、`>=`、`!=`、`==`来说，当左右任一操作数为`tensor`类型时，左右操作数的值不可同时为`bool`。
- 对于`==`来说，当左右操作数都为`number`类型时，支持左右操作数同时为`bool`，不支持只有一个操作数为`bool`。
- 对于`!=`、`==`来说除`mstype`外，其他取值均可和`None`进行比较来判空。