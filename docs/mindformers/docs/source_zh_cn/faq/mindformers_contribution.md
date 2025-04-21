# MindSpore Transformers贡献指南

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_zh_cn/faq/mindformers_contribution.md)

## 贡献代码至MindSpore Transformers

### 代码风格要求

请遵循此风格，以便MindSpore Transformers审查、维护和开发。

- 编码指南

  MindSpore Transformers社区使用`Python PEP 8` 编码风格。建议在IDE中安装以下插件，用于检查代码格式：`Lizard`、`ShellCheck` 和`PyLint`。

- 单元测试指南

  MindSpore Transformers社区使用Python单元测试框架pytest。注释名称需反映测试用例的设计意图。

- 重构指南

  我们鼓励开发人员重构我们的代码，以消除代码坏味道。所有代码都要符合编码风格和测试风格，重构代码也不例外。无注释的代码行（nloc）的Lizard阈值为100，圈复杂度（cnc）的阈值为20。当收到Lizard警告时，必须重构要合并的代码。

- 文档指南

  我们使用MarkdownLint来检查Markdown文档格式。基于默认配置修改了以下规则

    1. MD007（无序列表缩进）：参数indent设置为4，表示无序列表中的所有内容都需要缩进4个空格。
    2. MD009（行尾空格）：参数br_spaces设置为2，表示行尾可以有0或2个空格。
    3. MD029（有序列表的序列号）：参数style设置为ordered，表示升序。

### Fork-Pull 开发模型指导

- Fork MindSpore Transformers代码仓

  在提交代码至MindSpore Transformers项目之前，请确保已fork此项目到您自己的代码仓。MindSpore Transformers代码仓和您自己的代码仓之间可能会并行开发，请注意它们之间的一致性。

- 克隆远程代码仓

  如果您想将代码下载到本地计算机，最好使用git方法。

  ```shell
  # 在Gitee上克隆仓库
  git clone https://gitee.com/(insert_your_forked_repo)/mindformers.git
  ```

- 本地开发代码

  `dev`为开发分支，请从`dev`分支拉取最新代码进行开发。并在提交Pull Request时提交到`dev`分支。

  ```shell
  git checkout -b {新分支名称} origin/dev
  ```

- 提交PR到MindSpore Transformers代码仓

  在最后一步中，您需要在新分支和`MindSpore Transformers`主分支之间拉取比较请求。完成拉取请求后，`Jenkins CI`将自动设置，进行构建测试。PR应该尽快合并到上游dev分支中，以降低合并的风险。

  ```shell
  # 添加所有更改到暂存区
  git add

  # 查看更新状态
  git status

  # 提交更改，使用-m选项添加commit标题
  git commit -m "你的commit标题"

  # 添加commit的具体描述，使用-s选项添加签名，-amend选项修改最近一次提交
  git commit -s --amend

  # 推送更改到远程仓库的新分支
  git push origin {新分支名称}

  ```

### 文件及代码格式

若希望将自定义模型合入`MindSpore Transformers`代码仓库，需要注意几点：

1. 文件格式及位置要遵循规范。
2. 将新模型在代码中进行注册，以适配高阶接口使用。

#### 文件格式及位置

1. 模型代码文件统一放置于`research/{model_name}`文件夹下，格式如下:

    ```plaintext
    research/{model_name}
    ├── {model_name}
    | ├── {pretrain/finetune/predict}_{model_name}_{n}b.yaml
    | ├── convert_weight.py # Torch权重转MindSpore权重脚本（迁移模型需提供）
    | ├── convert_reversed.py # MindSpore权重转Torch权重脚本（迁移模型需提供）
    | ├── run_{model_name}.py # 运行代码文件
    | ├── {model_name}.py   # Model类代码文件
    | └── {model_name}_tokenizer.py # Tokenizer代码文件
    ```

2. 模型文档放置于同一`research/{model_name}`文件夹下。

## 提交PR的要求

### 只有一个commit

对于多commit的PR，请使用`squash`命令将多个commit合并为一个。
例如使用：

```shell
git rebase -i HEAD~3
```

可以看到:

```shell
pick 1234567 添加新功能A
pick 89abcdef 修复了功能A中的bug
pick 01234567 对功能A进行了一些优化
```

squash合并commit（可简化为 s, p, f 等简写）

```shell
pick 1234567 添加新功能A
squash 89abcdef 修复了功能A中的bug
squash 01234567 对功能A进行了一些优化
```

### PR描述

请使用以下md模板:

```markdown

### 相关的Issue

### 原因（目的、解决的问题等）

### 描述（做了什么，变更了什么）

### check list

#### 是否完成方案评审或问题根因分析（Y/N）

#### 是否完成了功能模块的UT/ST，并执行通过，附上结果（Y/N）

#### 是否涉及公共组件或对外接口修改，涉及时需给出修改范围和影响评估（Y/N）

#### 是否涉及资料修改，涉及时需同步修改（Y/N）

```

### 门禁要求

1. 提交PR需要[签署CLA](https://www.mindspore.cn/icla)。

2. 提交PR需要通过CI门禁检查，门禁失败修改代码后，需要在评论下评论`/retest`手动重启门禁检查。
