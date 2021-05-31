﻿# MindSpore文档

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
├───docs // 架构、网络和算子支持、编程指南等技术文档以及用于生成API的相关配置文件
|    |
|    ├───api_cpp // C++ API工程
|    |
|    ├───api_java // Java API工程
|    |
|    ├───api_python // Python API工程
|    |
|    ├───faq // 常见问题
|    |
|    ├───note // 设计和规格
|    |
|    └───programming_guide // 编程指南相关文档
│
├───install // 安装指南
│
├───lite // MindSpore Lite相关所有文档汇总及其链接  
│
├───resource // 资源相关文档
│
├───tools // 自动化工具
│
├───tutorials // 教程相关文档
|    |
|    ├───inference // 推理教程相关文档
|    |
|    ├───lite // 手机及IoT教程相关文档
|    |
|    ├───notebook // 体验式文档
|    |
|    ├───training // 训练教程相关文档
|    |
|    ├───tutorial_code // 教程对应样例代码
│
└───README_CN.md // Docs仓说明
```

## 文档构建

MindSpore的教程和API文档均可由[Sphinx](https://www.sphinx-doc.org/en/master/)工具生成。下面以Python API文档为例介绍具体步骤，操作前需完成MindSpore、MindSpore Hub和MindArmour的安装。

1. 下载MindSpore Docs仓代码。

   ```shell
   git clone https://gitee.com/mindspore/docs.git
   ```

2. 进入api_python目录，安装该目录下`requirements.txt`文件中的依赖项。

   ```shell
   cd docs/api_python
   pip install -r requirements.txt
   ```

3. 在api_python目录下执行如下命令，完成后会新建`build_zh_cn/html`目录，该目录中存放了生成后的文档网页，打开`build_zh_cn/html/index.html`即可查看API文档内容。

   ```shell
   make html
   ```

> - 构建[训练教程](https://gitee.com/mindspore/docs/tree/master/tutorials/training)和[编程指南](https://gitee.com/mindspore/docs/tree/master/docs/programming_guide)时还需安装[pandoc](https://pandoc.org/)，下载和安装pandoc请参考<https://pandoc.org/installing.html>。
>
> - 构建Python API时，如仅需生成MindSpore API，请先修改`source_zh_cn/conf.py`文件，注释`import mindspore_hub`和`import mindarmour`语句后，再执行以上步骤3。
>
> - 构建C++ API时，需要先克隆`mindspore`仓并安装doxygen，并加入环境变量`MS_PATH`：
>
>   ```bash
>   git clone https://gitee.com/mindspore/mindspore.git {MS_REPO PATH}
>   sudo apt install doxygen
>   export MS_PATH={MS_REPO PATH}
>   ```
>
>   其中`{MS_REPO PATH}`为克隆的`mindspore`仓路径。

## 版权

- [Apache License 2.0](LICENSE)
- [Creative Commons License version 4.0](LICENSE-CC-BY-4.0)
