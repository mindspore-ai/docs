# 副作用

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/design/side_effect.md" target="_blank">
<img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png">
</a>

## 概念

### 纯函数

函数的返回值只依赖函数的实参，且没有副作用的函数是纯函数。
纯函数更贴近数学意义上的函数：对于相同的输入参数，总能得到相同的返回值。
如果程序只包含纯函数，求值顺序不会影响程序结果。
比如下面的代码里，假设`add`是纯函数，那么`a`和 `b`的求值顺序不会影响`c`的结果：

```python
    a = add(1, 2)
    b = add(3, 4)
    c = add(a, b)
```

### 副作用

如果一个函数会改变外部状态，那么这个函数就具有副作用。
或者说带副作用的函数，除了函数的返回值，还有其他可以观察到的作用发生。
比如：修改全局变量、修改引用类型参数的值、执行输入输出操作、调用其他带有副作用的函数等。
当存在副作用时，程序的行为可能会因为求值顺序的不同而改变。
比如下面的代码里，假设`add`是纯函数、`assign`是有副作用的函数（会改变输入参数x），
那么`a`、`b`和 `c`的求值顺序不同就会导致`d`的结果不同：

```python
    a = add(1, x)
    b = assign(x, 100)  # side effect
    c = add(3, x)
    d = add(a, c)
```

由于存在副作用，上面的程序中`a`、`b`和 `c`应该严格按照它们在代码中的顺序求值，否则会产生不符合预期的结果。

## 设计

MindSpore采用的是一种基于图表示的函数式中间表示，
参考[MindIR](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/design/mindir.html)。
在概念上，MindIR中的函数都是纯函数，不存在副作用；
但MindSpore能够支持带副作用的计算模型，且提供带副作用的算子，比如会直接修改输入参数的优化器算子。
为了支持带副作用的算子和计算模型，MindSpore在编译模型的时候将代码中的副作用转换成了纯函数形式，
从而在保持MindIR的纯函数式语义不变的情况下，可以确保带副作用的计算按期望的顺序执行。

### 副作用转换为纯函数

为了能把带副作用的函数转换为纯函数形式，MindSpore将副作用函数所影响的外部状态看成是一个数据对象，
然后把函数对外部状态的修改转换为状态对象作为函数的输入，并将修改后的状态对象返回：

```python
    ret = func_with_side_effect(args)
```

转换为：

```python
    ret, state1 = pure_func(args, state0)
```

这里`pure_func`的返回值只依赖于输入参数，输入的状态`state0`不变，返回更新后的状态`state1`，因此可以看成是一个纯函数。

### 副作用的中间表示

由于MindIR的函数并不支持多个返回值，MindSpore引入了一个虚拟算子`UpdateState`，
将上面`pure_func`函数表达为类似下面形式的中间表示：

```python
    ret = pure_func(args, state0)
    state1 = UpdateState(state0, ret)
```

另外，为了确保读写顺序的正确性，MindSpore还引入了一个`Load`算子，
如果某个函数的输入是一个全局参数，则插入一个`Load`以确保函数读到正确的参数值。
例如下面代码里面的`add`需要读入一个全局参数`param`：

```python
    out = add(self.param, x)
```

MindSpore将其转换为类似下面形式的中间表示：

```python
    p = Load(self.param, state0)
    state1 = UpdateState(state0, p)
    out = add(p, x)
```

### 副作用分类

根据副作用所影响的外部状态类型不同，MindSpore将副作用分成三种类型：

1. 内存副作用：影响内存中的状态，比如修改全局变量、修改输入参数等；

2. 输入输出副作用：有输入输出操作，比如向控制台打印信息等；

3. 隐藏副作用：没有明显的外部状态改变，但实际存在隐藏状态改变。比如随机数生成算子，会影响随机数生成器的状态。

在MindSpore中，内存副作用和输入输出副作用分别用不同的状态对象表示，因此这两类副作用会体现为两条独立的执行序列；

隐藏副作用因为没有显式的外部状态对应，因此不会体现为独立的状态对象和执行序列，
但MindSpore内部会对其进行一些特殊处理，比如会阻止两个随机数生成算子的融合，以防止产生错误结果。

### 副作用算子标记

算子通过添加特定属性来标记是否具有副作用，MindSpore支持以下属性来标记算子的副作用：

- side_effect_mem 内存副作用
- side_effect_io 输入输出副作用
- side_effect_hidden 隐藏副作用

比如，将某个算子标记为具有内存副作用：

```python
    @prim_attr_register
    def __init__(self):
        ...
        self.add_prim_attr('side_effect_mem', True)
```

只有正确标识了副作用的算子，MindSpore才能确保其按期望的顺序执行。

## 相关场景

MindSpore能够自动识别代码中的副作用，并确保这些副作用按正确的顺序执行。
因此绝大多数情况下，模型开发者和使用者无需关注模型是否存在副作用以及如何确保正确的执行顺序。

### 算子开发

如果认为开发的算子具有副作用，需要通过算子属性正确标识该算子具有副作用，以及是哪种副作用，
否则使用了该算子的模型有可能因求值顺序未按预期执行导致错误的结果。

### 模型开发

通常情况，模型开发者不需要关注副作用，但理解副作用原理可能对代码执行顺序预期有帮助；
另外通过了解哪些算子具有副作用，也可以更好的作出算子选择。

### MindIR

如果模型具有副作用，在导出的MindIR里会存在`UpdateState`和`Load`节点，它们的作用是处理副作用和保序。
