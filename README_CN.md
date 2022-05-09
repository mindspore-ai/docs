# MindSpore文档

![MindSpore Logo](resource/MindSpore-logo.png)

[View English](./README.md)

## 简介

此工程提供MindSpore官方网站<https://www.mindspore.cn>所呈现安装指南、教程、文档的源文件以及API的相关配置。

## 贡献

我们非常欢迎您贡献文档！如果想要参与，请阅读[CONTRIBUTING_DOC_CN.md](./CONTRIBUTING_DOC_CN.md)，务必遵守文档写作规范，并按照流程规则提交，审核通过后，改动会在文档工程和官网中呈现。

同时，如果您对文档有任何意见或建议，请在Issues中提交。

## 目录结构说明

```text
docs
├───activity // 活动体验记录
|
├───docs // 架构、网络和算子支持等技术文档以及用于生成API的相关配置文件
|    |
|    ├───federated // MindSpore Federated（API工程、相关文档和常见问题）
|    |
|    ├───hub // MindSpore Hub（API工程和相关文档）
|    |
|    ├───lite // MindSpore Lite（API工程、相关文档和常见问题）
|    |
|    ├───mindarmour // MindArmour（API工程、相关文档和常见问题）
|    |
|    ├───mindinsight // MindInsight（相关文档和常见问题）
|    |
|    ├───mindquantum // MindQuantum（API工程和相关文档）
|    |
|    ├───mindspore // MindSpore（API工程、常见问题、迁移指南、说明）
|    |
|    ├───notebook // 体验式文档
|    |
|    ├───probability // MindSpore Probability（API工程和相关文档）
|    |
|    ├───sample_code // 文档对应样例代码
|    |
|    └───serving // MindSpore Serving（API工程、相关文档和常见问题）
|
│───install // 安装指南
|
│───resource // 资源相关文档
|
│───tools // 自动化工具
|
│───tutorials // MindSpore教程相关文档
|
└───README_CN.md // Docs仓说明
```

## 文档构建

MindSpore的教程和API文档均可由[Sphinx](https://www.sphinx-doc.org/en/master/)工具生成，构建MindSpore、MindSpore Hub、MindArmour或MindQuantum的API文档之前需完成对应模块的安装。下面以MindSpore Python API文档为例介绍具体步骤，操作前需完成MindSpore的安装。

1. 使用pip安裝MindSpore模块，API文档需要根据安装后的MindSpore模块生成，参考[安装](https://www.mindspore.cn/install)。

   ```bash
   pip install mindspore-1.5.0-cp37-cp37m-linux_x86_64.whl
   ```

2. 下载MindSpore Docs仓代码。

   ```bash
   git clone https://gitee.com/mindspore/docs.git
   ```

3. 进入api目录，安装该目录下`requirements.txt`文件中的依赖项。

   ```bash
   cd docs/mindspore
   pip install -r requirements.txt
   ```

4. 在api目录下执行如下命令，完成后会新建`build_zh_cn/html`目录，该目录中存放了生成后的文档网页，打开`build_zh_cn/html/index.html`即可查看API文档内容。

   ```bash
   make html
   ```

5. 构建MindSpore、MindInsight、MindQuantum等不同仓的API时，由于会使用到对应不同仓的一些资源文件，需要先克隆对应仓，并配置环境变量，给出以下配置列表供使用：

   | 对应API的生成 | 环境变量 | 仓库链接 | 仓库名 |
   | ---- | ---- | ---- | ---- |
   | MindSpore/ MindSpore Lite | MS_PATH | https://gitee.com/mindspore/mindspore.git | mindspore |
   | MindInsight | MI_PATH | https://gitee.com/mindspore/mindinsight.git | mindinsight |
   | MindArmour | MA_PATH | https://gitee.com/mindspore/mindarmour.git | mindarmour |
   | MindQuantum | MQ_PATH | https://gitee.com/mindspore/mindquantum.git | mindquantum |
   | Reinforcement | RM_PATH | https://gitee.com/mindspore/reinforcement.git | reinforcement |
   | Serving | SV_PATH | https://gitee.com/mindspore/serving.git | serving |
   | Vision | VS_PATH | https://gitee.com/mindspore/vision.git | vision |
   | Hub | HB_PATH | https://gitee.com/mindspore/hub.git | hub |

   克隆仓库以及设置环境变量的代码如下：

   ```bash
   git clone 仓库链接
   export 环境变量 = 对应克隆仓在本地的路径
   ```

   注意：

   - 构建Lite的API时，还需要安装Doxygen：

      ```bash
      sudo apt install doxygen
      ```

   - 构建[MindSpore教程](https://gitee.com/mindspore/docs/tree/master/tutorials)、[MindSpore文档](https://gitee.com/mindspore/docs/tree/master/docs/mindspore)、[深度概率编程文档](https://gitee.com/mindspore/docs/tree/master/docs/probability/docs)和[MindQuantum文档](https://gitee.com/mindspore/docs/tree/master/docs/mindquantum/docs)时还需安装[pandoc](https://pandoc.org/)，下载和安装pandoc请参考<https://pandoc.org/installing.html>。

## 版权

- [Apache License 2.0](LICENSE)
- [Creative Commons License version 4.0](LICENSE-CC-BY-4.0)
