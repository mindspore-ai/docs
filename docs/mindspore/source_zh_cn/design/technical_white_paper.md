# 技术白皮书

`Ascend` `GPU` `CPU` `设计`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/design/technical_white_paper.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 引言

深度学习研究和应用在近几十年得到了爆炸式的发展，掀起了人工智能的第三次浪潮，并且在图像识别、语音识别与合成、无人驾驶、机器视觉等方面取得了巨大的成功。这也对算法的应用以及依赖的框架有了更高级的要求。深度学习框架的不断发展使得在大型数据集上训练神经网络模型时，可以方便地使用大量的计算资源。

深度学习是使用多层结构从原始数据中自动学习并提取高层次特征的一类机器学习算法。通常，从原始数据中提取高层次、抽象的特征是非常困难的。目前有两种主流的深度学习框架：一种是在执行之前构造一个静态图，定义所有操作和网络结构，典型代表是TensorFlow，这种方法以牺牲易用性为代价，来提高训练期间的性能；另一种是立即执行的动态图计算，典型代表是PyTorch。通过比较可以发现，动态图更灵活、更易调试，但会牺牲性能。因此，现有深度学习框架难以同时满足易开发、高效执行的要求。

## 简介

MindSpore作为新一代深度学习框架，是源于全产业的最佳实践，最佳匹配昇腾处理器算力，支持终端、边缘、云全场景灵活部署，开创全新的AI编程范式，降低AI开发门槛。MindSpore是一种全新的深度学习计算框架，旨在实现易开发、高效执行、全场景覆盖三大目标。为了实现易开发的目标，MindSpore采用基于源码转换（Source Code Transformation，SCT）的自动微分（Automatic Differentiation，AD）机制，该机制可以用控制流表示复杂的组合。函数被转换成函数中间表达（Intermediate Representation，IR），中间表达构造出一个能够在不同设备上解析和执行的计算图。在执行前，计算图上应用了多种软硬件协同优化技术，以提升端、边、云等不同场景下的性能和效率。MindSpore支持动态图，更易于检查运行模式。由于采用了基于源码转换的自动微分机制，所以动态图和静态图之间的模式切换非常简单。为了在大型数据集上有效训练大模型，通过高级手动配置策略，MindSpore可以支持数据并行、模型并行和混合并行训练，具有很强的灵活性。此外，MindSpore还有“自动并行”能力，它通过在庞大的策略空间中进行高效搜索来找到一种快速的并行策略。MindSpore框架的具体优势，请查看详细介绍。

[查看技术白皮书](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com:443/white_paper/MindSpore_white_paperV1.1.pdf)
