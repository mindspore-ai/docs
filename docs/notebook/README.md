# MindSpore的教程体验

## 环境配置

### Windows和Linux系统配置方法

- 系统版本：Windows 10，Ubuntu 16.04及以上

- 软件配置：[Anaconda](https://www.anaconda.com/products/individual)，Jupyter Notebook

- 语言环境：Python3.7.X 推荐 Python3.7.5

- MindSpore 下载地址：[MindSpore官网下载](https://www.mindspore.cn/versions)，使用Windows系统用户选择Windows-X86版本，使用Linux系统用户选择Ubuntu-X86版本

> MindSpore的[具体安装教程](https://www.mindspore.cn/install/)

### Jupyter Notebook切换conda环境（Kernel Change）的配置方法

- 首先，增加Jupyter Notebook切换conda环境功能（Kernel Change）

  启动Anaconda Prompt，输入命令：

    ```bash
    conda install nb_conda
    ```

    > 建议在base环境操作上述命令。

  执行完毕，重启Jupyter Notebook即可完成功能添加。

- 然后，添加conda环境到Jypyter Notebook的Kernel Change中。

  1. 新建一个conda环境，启动Anaconda Prompt，输入命令：

      ```bash
      conda create -n {env_name} python=3.7.5
      ```

      > env_name可以按照自己想要的环境名称自行命名。

  2. 激活新环境，输入命令：

      ```bash
      conda activate {env_name}
      ```

  3. 安装ipykernel，输入命令：

      ```bash
      conda install -n {env_name} ipykernel
      ```

      > 如果添加已有环境，只需执行安装ipykernel操作即可。

  执行完毕后，刷新Jupyter notebook页面点击Kernel下拉，选择Kernel Change，就能选择新添加的conda环境。

## notebook说明

| 教&nbsp;&nbsp;程&nbsp;&nbsp;类&nbsp;&nbsp;别               | 教&nbsp;&nbsp;程&nbsp;&nbsp;名&nbsp;&nbsp;称                    | 文&nbsp;&nbsp;件&nbsp;&nbsp;名&nbsp;&nbsp;称       |  内&nbsp;&nbsp;容&nbsp;&nbsp;描&nbsp;&nbsp;述
| :-----------               | :-----------   | :-------              |:------
|  快速入门        | 手写数字分类识别入门体验教程           |   [mindspore_quick_start.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/beginner/quick_start.ipynb)     | - CPU平台下从数据集到模型验证的全过程解读 <br/> - 体验教程中各功能模块的使用说明 <br/> - 数据集图形化展示 <br/> - 了解LeNet5具体结构和参数作用 <br/> - 学习使用自定义回调函数 <br/> - loss值与训练步数的变化图 <br/> - 模型精度与训练步数的变化图 <br/> -  使用模型应用到手写图片的预测与分类上
| 快速入门         | 线性拟合         | [mindspore_linear_regression.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/linear_regression.ipynb)       | - 了解线性拟合的算法原理<br/> - 了解在MindSpore中如何实现线性拟合的算法原理 <br/> - 学习使用MindSpore实现AI训练中的正向传播和方向传播<br/> - 可视化线性函数拟合数据的全过程。
| 基础使用         | 加载图像数据集        | [mindspore_load_dataset_image.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/load_dataset_image.ipynb)           | - 学习加载图像数据集 <br/> - 学习处理图像数据集<br/> - 学习增强图像数据集
| 基础使用         | 加载文本数据集        | [mindspore_load_dataset_text.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/load_dataset_text.ipynb)           | - 学习加载文本数据集 <br/> - 学习处理文本数据集<br/> - 学习文本数据集分词
| 基础使用         |  保存模型   |     [mindspore_save_model.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/notebook/mindspore_save_model.ipynb)          | - 了解不同平台用于训练的模型类型<br/> - 学习如何用不同策略保存训练模型<br/> - 学习如何将模型导出为不同的文件类型，用于不同平台上的训练
| 基础使用         | 加载模型用于推理或迁移学习         | [mindspore_load_model_for_inference_and_transfer.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/notebook/mindspore_load_model_for_inference_and_transfer.ipynb)       | - 了解预训练模型的方法<br/> - 学习在本地加载已有的模型进行推理的方法<br/> - 学习在本地加载模型并进行迁移学习的方法
| 处理数据         | 转换数据集为MindRecord        | [mindspore_convert_dataset.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/convert_dataset.ipynb)           | - 展示将数据集转换为MindRecord <br/> - 展示读取MindRecord数据集
| 数据处理        | 优化数据准备的性能         | [mindspore_optimize_data_processing.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/optimize_data_processing.ipynb)       | - 数据加载性能优化<br/> - shuffle性能优化<br/> - 数据增强性能优化<br/> - 性能优化方案总结
| 应用实践        | 自然语言处理应用         |  [mindspore_nlp_application.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/notebook/mindspore_nlp_application.ipynb)         | - 展示MindSpore在自然语言处理的应用<br/> - 展示自然语言处理中数据集特定的预处理方法<br/> - 展示如何定义基于LSTM的SentimentNet网络
| 应用实践         | 计算机视觉应用     | [mindspore_computer_vision_application.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/notebook/mindspore_computer_vision_application.ipynb)       | - 学习MindSpore卷积神经网络在计算机视觉应用的过程 <br/> - 学习下载CIFAR-10数据集，搭建运行环境<br/>- 学习使用ResNet-50构建卷积神经网络<br/> - 学习使用Momentum和SoftmaxCrossEntropyWithLogits构建优化器和损失函数<br/> - 学习调试参数训练模型，判断模型精度
| 调试网络         | 模型的训练及验证同步方法         | [mindspore_evaluate_the_model_during_training.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/notebook/mindspore_evaluate_the_model_during_training.ipynb)       | - 了解模型训练和验证同步进行的方法<br/> - 学习同步训练和验证中参数设置方法<br/> - 利用绘图函数从保存的模型中挑选出最优模型
| 调试网络         | 使用PyNative进行神经网络的训练调试体验          | [mindspore_debugging_in_pynative_mode.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/notebook/mindspore_debugging_in_pynative_mode.ipynb)      | - GPU平台下从数据集获取单个数据进行单个step训练的数据变化全过程解读 <br/> - 了解PyNative模式下的调试方法 <br/> - 图片数据在训练过程中的变化情况的图形展示 <br/> - 了解构建权重梯度计算函数的方法 <br/> - 展示1个step过程中权重的变化及数据展示
| 调试网络         | 自定义调试信息体验文档         | [mindspore_custom_debugging_info.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/notebook/mindspore_custom_debugging_info.ipynb)       | - 了解MindSpore的自定义调试算子 <br/> - 学习使用自定义调试算子Callback设置定时训练<br/>- 学习设置metrics算子输出相对应的模型精度信息<br/> - 学习设置日志环境变量来控制glog输出日志
| 调试网络        |  MindInsight的溯源分析和对比分析            |  [mindspore_lineage_and_scalars_comparison.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/notebook/mindinsight/mindspore_lineage_and_scalars_comparison.ipynb)       | - 了解MindSpore中训练数据的采集及展示  <br/> - 学习使用回调函数SummaryCollector进行数据采集 <br/> - 使用MindInsight进行数据可视化 <br/> - 了解数据溯源和模型溯源的使用方法 <br/> - 了解对比分析的使用方法
| 调试网络          | MindInsight训练看板         | [mindinsight_dashboard.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/notebook/mindinsight/mindinsight_dashboard.ipynb)       | - 了解完整的MindSpore深度学习及MindInsight可视化展示的过程 <br/> - 学习使用MindInsight对训练过程中标量、直方图、图像、计算图、数据图和张量信息进行可视化展示<br/> - 学习使用Summary算子记录标量、直方图、图像、计算图、数据图和张量信息
| 调试网络          | 应用自动数据增强         | [mindspore_enable_auto_augmentation.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/notebook/mindspore_enable_auto_augmentation.ipynb)       | - 了解自动数据增强原理 <br/> - 在CIFAR-10数据集上实现自动数据增强
| 优化训练性能           | 应用梯度累积算法         | [mindspore_apply_gradient_accumulation.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/notebook/mindspore_apply_gradient_accumulation.ipynb)       | - 了解梯度累积训练算法的原理  <br/> - 学习在MindSpore中自定义训练函数，并在其中实现前向传播和反向传播的计算过程构建，实现梯度累积计算 <br/> - 了解梯度累积训练对模型训练的影响
| 模型安全和隐私        | 模型安全         | [mindspore_improve_model_security_nad.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/notebook/mindspore_improve_model_security_nad.ipynb)       | - 了解AI算法的安全威胁的概念和影响<br/> - 介绍MindArmour提供的模型安全防护手段<br/> - 学习如何模拟攻击训练模型<br/> - 学习针对被攻击模型进行对抗性防御
| 编程指南        | MindSpore API概述     | [mindspore_api_structure.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/api_structure.ipynb)       | - 了解MindSpore的总体架构<br/> - 介绍MindSpore的设计理念<br/> - 介绍MindSpore的层次结构
| 编程指南        | dtype                 | [mindspore_dtype.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/dtype.ipynb)       | - 了解MindSpore支持的数据类型<br/> - 介绍MindSpore数据类型转换的接口
| 编程指南        | Tensor     | [mindspore_tensor.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/tensor.ipynb)       | - 了解tensor的构造<br/> - 介绍tensor的属性
| 编程指南        | Parameter     | [mindspore_parameter.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/parameter.ipynb)       | - 介绍Parameter的初始化<br/> - 介绍Parameter的属性<br/> - 介绍Parameter的方法
| 编程指南        | Cell构建及其子类     | [mindspore_cell.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/cell.ipynb)       | - 介绍Cell的关键成员函数<br/> - 介绍nn模块与ops模块的关系<br/> - 了解模型层<br/> - 了解MindSpore的损失函数<br/> - 学习MindSpore优化算法<br/> - 学习构建自定义网络
| 编程指南        | 数据集加载     | [mindspore_dataset_loading.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/dataset_loading.ipynb)       | - 了解常用数据集加载<br/> - 学习特定格式数据集加载<br/> - 学习自定义数据集加载
| 编程指南        | 采样器     | [mindspore_sampler.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/sampler.ipynb)       | - 介绍MindSpore采样器<br/> - 学习如何自定义采样器
| 编程指南        | 数据处理     | [mindspore_pipeline_common.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/pipeline_common.ipynb)       | - 学习MindSpore数据处理算子的使用方法
| 编程指南        | 数据增强     | [mindspore_augmentation.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/augmentation.ipynb)       | - 介绍MindSpore提供的数据增强操作<br/> - 介绍c_transforms使用方法<br/> - 介绍py_transforms使用方法
| 编程指南        | 分词器     | [mindspore_tokenizer.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/tokenizer.ipynb)       | - 介绍MindSpore提供的分词器
| 编程指南        | MindSpore数据格式转换     | [mindspore_dataset_conversion.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/dataset_conversion.ipynb)       | - 介绍非标准数据集转换为MindRecord<br/> - 介绍常用数据集转换为MindRecord
| 编程指南        | 自动数据增强     | [auto_augmentation.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/auto_augmentation.ipynb)       | - 介绍基于概率的自动数据增强<br/> - 介绍基于回调参数的自动数据增强
| 编程指南        | 运行管理     | [mindspore_context.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/context.ipynb)       | - 介绍执行模式的选择和切换<br/> - 介绍硬件管理的方式<br/> - 介绍分布式管理的用法<br/> - 介绍维测管理的使用方式
| 编程指南        | 运行方式     | [mindspore_run.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/run.ipynb)       | - 介绍单个算子的执行<br/> - 介绍普通函数的执行<br/> - 介绍网络模型的执行
| 编程指南        | 深度概率编程库     | [mindspore_probability.ipynb](https://gitee.com/mindspore/docs/blob/master/docs/probability/docs/source_zh_cn/probability.ipynb)       | - 介绍概率分布类<br/> - 介绍概率分布映射<br/> - 学习深度概率网络构建<br/> - 学习概率推断算法<br/>  - 学习贝叶斯网络的构建及转换<br/> - 学习MindSpore中贝叶斯工具箱的使用方法
