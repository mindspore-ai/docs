# 复现算法实现

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/reproducing_algorithm.md)

## 获取参考代码

我们拿到一篇论文，需要在MindSpore上进行迁移实现时，优先需要找到在其他框架已经实现好的参考代码，原则上这个参考代码需要符合以下要求中的至少一项：

1. 论文原作者开源的实现；
2. 大众普遍认可的实现(star数，fork数较多)；
3. 比较新的代码，有开发者对代码进行维护；
4. 优先考虑PyTorch的参考代码。

如果参考项目中结果无法复现或者缺乏版本信息，可查看项目issue获取信息；

如果是全新的论文，无可参考实现，请参考[MindSpore网络搭建](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/model_development/model_development.html)进行开发。

## 分析算法及网络结构

在阅读论文及参考代码时，首先需要分析网络结构，用以组织代码编写。比如下面是YOLOX的大致网络结构：

| 模块 | 实现 |
| ---- | ---- |
| backbone | CSPDarknet(s,m,l,x等) |
| neck | FPN |
| head | Decoupled Head |

其次需要分析迁移算法的创新点，记录在训练过程中使用了哪些trick，如数据处理加了哪些数据增强，是否有shuffle，使用了什么优化器，学习率衰减策略，参数初始化方式等。可以整理一个checklist，在分析过程中可以填写相应项来记录。

比如这里记录了YOLOX网络在训练时使用的一些trick：

<table>
    <tr>
        <th>trick</th>
        <th>记录</th>
   </tr>
    <tr>
        <td rowspan="2">数据增强</td>
        <td >mosaic，包含随机缩放，随机剪裁，随机排布 </td>
    </tr>
    <tr>
        <td >MixUp</td>
    </tr>
    <tr>
        <td >学习率衰减策略</td>
        <td >多种衰减方式供选择，默认使用cos学习率衰减</td>
    </tr>
    <tr>
        <td >优化器参数</td>
        <td >带动量SGD momentum=0.9，nesterov=True，无weight decay</td>
    </tr>
    <tr>
        <td >训练参数</td>
        <td >epoch：300；batchsize：8</td>
    </tr>
    <tr>
        <td >网络结构优化点</td>
        <td >Decoupled Head；Anchor Free；SimOTA</td>
    </tr>
    <tr>
        <td >训练流程优化点</td>
        <td >EMA；后15epoch不做数据增强；混合精度</td>
    </tr>
</table>

**注意，以复现代码中使用的trick为主，有些论文里提到的不一定有用。**

此外，需要判断论文是否能通过在MindSpore已有模型上做少量修改来实现，若是，可以在已有模型的基础上进行开发，这样能极大的减少开发的工作量。比如WGAN-PG可以基于WGAN进行开发。
[MindSpore models](https://gitee.com/mindspore/models)是MindSpore的模型仓库，当前已经覆盖了机器视觉、自然语言处理、语音、推荐系统等多个领域的主流模型，可以从中查找是否有需要的模型。

## 复现论文实现

获取到参考代码后，需要复现下参考实现的精度，获取参考实现的性能数据。这样做有几点好处：

1. 提前识别一些问题：

    - 判断参考代码使用的三方库是否有版本依赖，提前识别版本适配问题；
    - 判断数据集是否能获取的到，有的数据集是私有的或者原作者在公开数据集上加了自己的部分数据集，在复现参考实现阶段就可以发现这种问题；
    - 参考实现是否能复现论文精度，有的官方的参考实现也不一定能复现论文的精度，当出现这种情况时要及时发现问题，更换参考实现或者调整精度基线。

2. 获取一些参考数据作为MindSpore迁移过程的参考：

    - 获取loss下降趋势，帮助验证MindSpore上训练收敛趋势是否ok；
    - 获取参数文件，用于进行转换，进行推理验证，详细过程参考[推理及训练流程](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/model_development/training_and_evaluation.html)；
    - 获取性能基线，在做性能优化时有一个基础目标，如需做性能优化，请参考[调试调优](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/debug_and_tune.html)。
