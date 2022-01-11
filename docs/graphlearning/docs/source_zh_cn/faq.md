# FAQ

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/graphlearning/docs/source_zh_cn/faq.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

<font size=3>**Q: 命令行执行GNNCell报错`OSError: could not get source code`怎么办？**</font>

<font size=3>**A:** MindSpore Graph
Learning源到源翻译解析以点为中心的编程代码，中间调用了inspect来获取翻译module的源码，需要将模型定义代码放到Python文件中，否则会报错找不到源码。

<font size=3>**Q:
执行GNNCell报错`AttributeError: None of backend from {mindspore} is identified. Backend must be imported as a global variable. `
怎么办？**</font>

<font size=3>**A:** MindSpore Graph Learning通过源到源翻译解析以点为中心的编程代码，在GNNCell定义的文件根据全局变量获取网络执行后端，需要在GNNCell定义文件头部import
mindspore，否则会报错找不到后端。

<font size=3>**Q: 调用图聚合接口'sum、avg、max、min'
等时`TypeInferenceError: Line 6: Built-in agg func "avg" only takes expr of EDGE or SRC type. Got None.`怎么办？**</font>

<font size=3>**A:** MindSpore Graph Learning前端提供的聚合接口均为针对图节点的操作，在源到源翻译过程会判断聚合接口的输入是否为图中的边或节点，否则报错找不到合乎规则的输入类型。

<font size=3>**Q: 调用图接口'dot'时`RuntimeError: The 'mul' operation does not support the type.`怎么办？**</font>

<font size=3>**A:** MindSpore Graph
Learning前端提供的dot接口为针对图节点的点乘操作，后端包含特征乘和聚合加两步，前端翻译过程不涉及编译无法判断输入数据类型，输入类型必须符合后端mul算子的类型要求，否则会报错类型不支持。

<font size=3>**Q: 调用图接口'topk_nodes,
topk_edges'时`TypeError: For 'tensor getitem', the types only support 'Slice', 'Ellipsis', 'None', 'Tensor', 'int', 'List', 'Tuple', 'bool', but got String.`
怎么办？**</font>

<font size=3>**A:** MindSpore Graph
Learning前端提供的topk_nodes接口为针对图节点/边特征排序取k个节点/边的操作，后端包含获取节点/边特征、排序sort和slice取k个三步，前端翻译过程不涉及编译无法判断输入数据类型，输入类型必须符合sort和slice算子的排序维度sortby和取值范围k的类型要求，否则会报错类型不支持。