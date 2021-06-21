# 前端编译

`Linux` `Windows` `Ascend` `GPU` `CPU` `环境准备` `初级` `中级`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_zh_cn/frontend_compile.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

<font size=3>**Q：运行时报错“Create python object \`<class 'mindspore.common.tensor.Tensor'>\` failed, only support create Cell or Primitive object.”怎么办？**</font>

A：当前图模式不支持在网络里构造`Tensor`，即不支持语法`x = Tensor(args...)`。

如果是常量`Tensor`，请在`__init__`函数中定义。如果不是常量`Tensor`，可以通过`@constexpr`装饰器修饰函数，在函数里生成`Tensor`。

关于`@constexpr`的用法可参考：<https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/ops/mindspore.ops.constexpr.html>。

对于网络中需要用到的常量`Tensor`，可以作为网络的属性，在`init`的时候定义，即`self.x = Tensor(args...)`，然后在`construct`里使用。

如下示例，通过`@constexpr`生成一个`shape = (3, 4), dtype = int64`的`Tensor`。

```python
@constexpr
def generate_tensor():
    return Tensor(np.ones((3, 4)))
```

<br/>

<font size=3>**Q：运行时报错“'self.xx' should be defined in the class '__init__' function.”怎么办？**</font>

A：如果在`construct`函数里，想对类成员`self.xx`赋值，那么`self.xx`必须已经在`__init__`函数中被定义为[`Parameter`](<https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.html?highlight=parameter#mindspore.Parameter>)类型，其他类型则不支持。局部变量`xx`不受这个限制。

<br/>

<font size=3>**Q：运行时报错“This comparator 'AnyValue' is not supported. For statement 'is', only support compare with 'None', 'False' or 'True'”怎么办？**</font>

A：对于语法`is` 或 `is not`而言，当前`MindSpore`仅支持与`True`、`False`和`None`的比较。暂不支持其他类型，如字符串等。

<br/>

<font size=3>**Q：运行时报错“MindSpore does not support comparison with operators more than one now, ops size =2”怎么办？**</font>

A：对于比较语句，`MindSpore`最多支持一个操作数。例如不支持语句`1 < x < 3`，请使用`1 < x and x < 3`的方式代替。

<br/>

<font size=3>**Q：运行时报错“TypeError: The function construct need 1 positional argument and 0 default argument, but provided 2”怎么办？**</font>

A：网络的实例被调用时，会执行`construct`方法，然后会检查`construct`方法需要的参数个数和实际传入的参数个数，如果不一致则会抛出以上异常。
请检查脚本中调用网络实例时传入的参数个数，和定义的网络中`construct`函数需要的参数个数是否一致。

<br/>
