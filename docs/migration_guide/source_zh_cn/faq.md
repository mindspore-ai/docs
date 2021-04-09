# 常见问题

- 网络脚本分析

    Q：在迁移的构成中发现其他框架中的网络迁移到MindSpore中时，并没有对应的MindSpore支持的算子怎么办？

    A：可参考网络脚本分析的缺失算子处理策略：<https://www.mindspore.cn/doc/migration_guide/zh-CN/master/script_analysis.html#id4>。

- 网络调试

    Q：网络调试过程中，算子执行报错；在PyNative模式下能跑通，但Graph模式下报错；分布式并行训练脚本配置错误等问题怎么处理？

    A：可参考《网络调试》中的常见方法：<https://www.mindspore.cn/doc/migration_guide/zh-CN/master/neural_network_debug.html#id4>。

- 模型精度调优

    Q：模型精度调优可能会遇到哪些问题，如何处理？

    A：模型精度调优可以参考《[MindSpore模型精度调优实战](https://www.mindspore.com/doc/migration_guide/zh-CN/master/accuracy_optimization.html#mindspore)》系列；调优过程中需要使用到的辅助工具，超参调整，网络结构调整等，可以参考模型精度调优中的[参考文档](https://www.mindspore.cn/doc/migration_guide/zh-CN/master/accuracy_optimization.html#id2)。

- 使用Profiler性能调试工具

    Q：启动Profiler失败怎么解决？

    A：可参考《性能调试工具》中的常见问题处理：<https://www.mindspore.cn/doc/migration_guide/zh-CN/master/performance_optimization.html#id6>。

- 推理迁移

    Q：迁移到MindSore后如何创建推理服务？

    A：在MindSpore中创建推理服务，可参考MindSpore Serving类：<https://www.mindspore.cn/doc/faq/zh-CN/master/mindspore_serving.html>。

- 网络迁移调试实例

    Q：在进行网络迁移调试过程中，可能碰到哪些问题？如何处理？

    A：可以参考网络迁移调试实例的常见问题及相应的优化方法：<https://www.mindspore.cn/doc/migration_guide/zh-CN/master/sample_code.html#id26>。

- 迁移案例

    Q：如何修改迁移后脚本的批次大小（Batch size）、句子长度（Sequence length）等尺寸（shape）规格，以实现模型可支持任意尺寸的数据推理、训练？

    A：迁移后脚本存在shape限制，通常是由于Reshape算子导致，或其他涉及张量排布变化的算子导致。以上述Bert迁移为例，首先创建两个全局变量，表示预期的批次大小、句子长度，而后修改Reshape操作的目标尺寸，替换成相应的批次大小、句子长度的全局变量即可。

    Q：生成后的脚本中类名的定义不符合开发者的习惯，如class Module0(nn.Cell)，人工修改是否会影响转换后的权重加载？

    A：权重的加载仅与变量名、类结构有关，因此类名可以修改，不影响权重加载。若需要调整类的结构，则相应的权重命名需要同步修改以适应迁移后模型的结构。
