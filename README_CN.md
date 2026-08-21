# MindSpore 文档

![MindSpore Logo](resource/MindSpore-logo.png)

[View English](./README.md)

## 简介

此工程提供 MindSpore 官方网站 <https://www.mindspore.cn> 所呈现的安装指南、教程、文档的源文件以及 API 的相关配置。

## 贡献

我们非常欢迎您贡献文档！如果想要参与，请阅读 [CONTRIBUTING_DOC_CN.md](./CONTRIBUTING_DOC_CN.md)，务必遵守文档写作规范，并按照流程规则提交。审核通过后，改动会在文档工程和官网中呈现。

同时，如果您对文档有任何意见或建议，请在 Issues 中提交。

## 目录结构说明

```text
docs
│
├───docs // 设计、规格、FAQ 等技术文档，以及用于生成 API 的相关配置文件
│    │
│    ├───golden_stick // MindSpore Golden Stick 文档
│    │
│    ├───lite // MindSpore Lite 文档
│    │
│    ├───mindarmour // MindSpore Armour 文档
│    │
│    ├───mindformers // MindSpore Transformers 文档
│    │
│    ├───mindquantum // MindSpore Quantum 文档
│    │
│    ├───mindscience // MindScience 文档
│    │
│    ├───mindspore // MindSpore 文档
│    │
│    ├───mindstudio // MindStudio 文档
│    │
│    ├───msadapter // MSAdapter 文档
│    │
│    ├───sample_code // 文档对应样例代码
│    │
│    └───vllm_mindspore // vLLM-MindSpore Plugin 文档
│
├───install // 安装指南
│
├───resource // 资源相关文档
│
├───templates // 文档模板和样例
│
├───tools // 自动化工具
│
├───tutorials // MindSpore 教程相关文档
│
├───CODEOWNERS // Maintainer 列表
│
├───CONTRIBUTING_DOC.md // 贡献文档英文版
│
├───CONTRIBUTING_DOC_CN.md // 贡献文档中文版
│
├───LICENSE // LICENSE 文档
│
├───NOTICE // NOTICE 文档
│
├───README.md // Docs 仓说明英文版
│
└───README_CN.md // Docs 仓说明中文版
```

## 文档构建

MindSpore 的教程和 API 文档均可由 [Sphinx](https://www.sphinx-doc.org/en/master/) 工具生成。构建 MindSpore 等各组件 API 文档之前，需完成对应模块的安装。下面以 MindSpore Python API 文档为例介绍具体步骤，操作前需完成 MindSpore 的安装。

1. 使用 pip 安装 MindSpore 模块，API 文档需要根据安装后的 MindSpore 模块生成，参考[安装](https://www.mindspore.cn/install)。

   ```bash
   pip install mindspore-*.*.*-cp312-cp312-linux_x86_64.whl
   ```

2. 下载 MindSpore Docs 仓代码。

   ```bash
   git clone https://atomgit.com/mindspore/docs.git
   ```

3. 进入 API 所在目录 `docs/mindspore`，安装该目录下 `requirements.txt` 文件中的依赖项。

   ```bash
   cd docs/mindspore
   pip install -r requirements.txt
   ```

4. 在 API 所在目录 `docs/mindspore` 下打开配置文件 `Makefile`，根据要生成文档的语言进行配置。其中 `SOURCEDIR` 指源文件夹，`BUILDDIR` 指构建完成后文档的文件夹名称。

   ```text
   SOURCEDIR     = source_zh_cn
   BUILDDIR      = build_zh_cn
   ```

   - 构建中文文档，将 `SOURCEDIR` 配置成 `source_zh_cn`，将 `BUILDDIR` 配置成 `build_zh_cn`。
   - 构建英文文档，将 `SOURCEDIR` 配置成 `source_en`，将 `BUILDDIR` 配置成 `build_en`。

5. 文件 `Makefile` 配置完成后，在 API 所在目录 `docs/mindspore` 下执行如下命令进行文档构建：

   ```bash
   make html
   ```

   - 中文构建：完成后会新建 `build_zh_cn/html` 目录，该目录中存放了生成后的中文文档网页，打开 `build_zh_cn/html/index.html` 即可查看 API 文档内容。
   - 英文构建：完成后会新建 `build_en/html` 目录，该目录中存放了生成后的英文文档网页，打开 `build_en/html/index.html` 即可查看 API 文档内容。

### 注意事项

1. 构建 MindSpore 等不同仓的 API 时，由于会使用到对应不同仓的一些资源文件，需要先克隆对应仓，并配置环境变量，给出以下配置列表供使用：

   | 对应 API 的生成 | 环境变量 | 仓库链接 | 仓库名 |
   | ---- | ---- | ---- | ---- |
   | MindSpore | MS_PATH | <https://atomgit.com/mindspore/mindspore.git> | mindspore |
   | MindSpore Lite | MSL_PATH | <https://atomgit.com/mindspore/mindspore-lite.git> | mindspore_lite |
   | MindSpore Transformers | MFM_PATH | <https://atomgit.com/mindspore/mindformers.git> | mindformers |
   | MindSpore Golden Stick | GS_PATH | <https://atomgit.com/mindspore/golden-stick.git> | golden_stick |
   | MindSpore Quantum | MQ_PATH | <https://atomgit.com/mindspore/mindquantum.git> | mindquantum |
   | MindScience | MSC_PATH | <https://atomgit.com/mindspore/mindscience.git> | mindscience |

   克隆仓库以及设置环境变量的代码如下：

   ```bash
   git clone 仓库链接
   export 环境变量=对应克隆仓在本地的路径
   ```

2. 构建 Lite 的 API 时，还需要安装 Doxygen，且要下载最新的 Lite tar 包，并将本地的包路径配置到 LITE_PACKAGE_PATH 环境变量：

   ```bash
   sudo apt install doxygen
   export LITE_PACKAGE_PATH=本地的Lite包路径
   ```

   其中，发布包需要同时包含端侧和云侧，并按以下方式存放：

   ```text
   LITE_PACKAGE_PATH
   ├───mindspore-lite-*.*.*-linux-x64.tar.gz
   │
   └───cloud_fusion
          │
          └───mindspore-lite-*.*.*-linux-x64.tar.gz
   ```

3. 构建 [MindSpore 教程](https://atomgit.com/mindspore/docs/tree/master/tutorials)、[MindSpore 文档](https://atomgit.com/mindspore/docs/tree/master/docs/mindspore)和 [MindQuantum 文档](https://atomgit.com/mindspore/docs/tree/master/docs/mindquantum/docs)时还需安装 [pandoc](https://pandoc.org/)，下载和安装 pandoc 请参考 <https://pandoc.org/installing.html>。

4. 从 2.10 版本开始使用 Python 3.12 构建文档，2.9 及以前版本使用 Python 3.9 构建文档。

## 版权

- [Apache License 2.0](LICENSE)
