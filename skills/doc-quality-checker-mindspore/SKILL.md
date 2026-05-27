---
name: doc-quality-checker-mindspore
description: |
  检查文档质量的工具。当用户提到检查文档质量、审查文档、文档检查、文档审查、lint文档，
  或者提供文档URL/PR链接/本地文件路径要求检查时触发。支持通用性检查、教程检查和API文档检查。
  输入类型：1) 可访问的HTML页面URL；2) AtomGit PR链接（分析PR中的文档diff）；
  3) 本地文件夹路径。自动根据内容判断执行 通用+教程 或 通用+API 的检查组合。
---

# 文档质量检查器

## 输入解析

解析用户提供的输入，判断输入类型并获取内容：

| 输入类型 | 识别方式 | 获取内容方法 |
|---------|---------|-------------|
| HTML页面URL | 以 `http://` 或 `https://` 开头且非Git链接 | 使用 `WebFetch` 工具获取页面内容 |
| AtomGit PR链接 | 包含 `atomgit.com` 的PR URL | 使用 AtomGit API + Token 获取PR内容 |
| 本地文件夹 | 本地路径 | 使用 `read` 工具读取文件内容 |

### PR链接处理流程

对于 AtomGit PR链接：

1. **获取Token** → 提示用户输入 AtomGit/Gitee 私人令牌（用于API认证）
2. 解析PR链接获取仓库信息（owner）、仓库名（repo）和PR编号
3. 调用脚本 `scripts/fetch_pr.ps1` 或 `scripts/fetch_pr.sh` 获取PR详情和diff内容：
    - 脚本路径：`scripts/fetch_pr.ps1`（Windows）或 `scripts/fetch_pr.sh`（Linux/Mac）
    - 使用 `PRIVATE-TOKEN` header 认证：`--header 'PRIVATE-TOKEN: {token}'`
    - API格式: `https://api.atomgit.com/api/v5/repos/{owner}/{repo}/pulls/{pr_number}`
4. 识别diff中的文档文件（.md, .rst, .txt等）
5. 分析文档变更内容

**脚本用法：**

```powershell
# Windows
.\scripts\fetch_pr.ps1 -Token <token> -Owner <owner> -Repo <repo> -PrNumber <pr_number>

# Linux/Mac
bash scripts/fetch_pr.sh <token> <owner> <repo> <pr_number>
```

**Token获取方式**：

- AtomGit: https://atomgit.com/user_settings/apitokens
- 需要创建私人令牌，勾选 `repo` 权限

## 检查类型判断

根据输入内容特征自动选择检查组合：

| 特征 | 检查组合 |
|-----|---------|
| 包含"安装"、"快速开始"、"教程"、"开发指南"等 | 通用性检查 + 教程检查 |
| 包含"API"、"参数"、"返回值"、"函数"等 | 通用性检查 + API文档检查 |
| 包含代码示例、函数签名、参数说明 | 通用性检查 + API文档检查 |
| 包含步骤说明、截图指引、操作流程 | 通用性检查 + 教程检查 |
| 无法判断时默认 | 通用性检查 + 教程检查 |

## 检查规则

### 1. 通用性检查（所有文档必须通过）

规则文件为`rules/doc_general_rules.md`，支持所有格式文件的内容检查。

### 2. 教程检查（教程类文档额外检查）

规则文件为`rules/doc_tutorials_rules.md`，支持Markdown、reStructuredText、Jupyter Notebook等格式文件的内容检查。

### 3. API文档检查（API类文档额外检查）

**重要**：API文档检查必须同时包含以下两部分，缺一不可：

1. **格式检查**：检查文档本身的格式和质量（如参数说明、返回值、异常等）
2. **一致性检查**：检查中文API文档与英文源文件（Python/C++/YAML/RST目录）的对应关系

根据输入文档的语言类型，选择对应的规则文件：

| 文档语言 | 规则文件 | 说明 |
|---------|---------|------|
| Python API中文文档（RST格式） | `rules/api_python_zh_rules.md` | Python类/函数RST文档 |
| Python API英文注释（Python Docstring或YAML格式） | `rules/api_python_en_rules.md` | 检查Python docstrings、YAML配置 |
| Python API中文与英文一致性 | `rules/api_python_consistency_rules.md` | Python API中文文档与英文源码的一致性检查 |
| C++ API中文文档（MD格式） | `rules/api_cpp_zh_rules.md` | C++类/函数MD文档 |
| C++ API英文注释 | `rules/api_cpp_en_rules.md` | 检查C++ `///` 注释 |

#### 检查组合逻辑

- **中文API文档（RST/MD格式）**：
    1. 应用对应规则文件检查格式：

        - Python类/函数RST文档：`rules/api_python_zh_rules.md`
        - C++类/函数MD文档：`rules/api_cpp_zh_rules.md`

    2. 识别对应的英文源文件（.py对应RST，.yaml/.yml对应RST，.h/.cpp对应MD）
    3. 应用 `rules/api_python_consistency_rules.md` 检查与英文的一致性

- **Python API英文注释**：
    1. 应用 `rules/api_python_en_rules.md` 检查注释格式
    2. （如需与中文对应）使用 `rules/api_python_consistency_rules.md` 检查与中文文档的一致性

- **C++ API英文注释**：
    1. 应用 `rules/api_cpp_en_rules.md` 检查注释格式
    2. （如需与中文对应）使用 `rules/api_python_consistency_rules.md` 检查与中文文档的一致性

- **AtomGit PR链接**：
    1. 获取PR中的所有文档修改（.rst, .py, .yaml, .md, .h, .cpp等）
    2. 建立中文文档与英文源码的关联
    3. 检查PR中是否有中文修改但无英文对应，或英文修改但无中文对应
    4. 应用 `rules/api_python_consistency_rules.md` 中的"PR配对检查"规则
    5. 对每个修改的中文RST/中文MD文件应用对应规则文件检查格式：

        - 中文RST文件：`rules/api_python_zh_rules.md`
        - 中文MD文件：`rules/api_cpp_zh_rules.md`

    6. 对每个修改的Python代码、YAML文件应用 `rules/api_python_en_rules.md` 检查格式

        - 对每个修改的C++头文件应用 `rules/api_cpp_en_rules.md` 检查格式

    7. 特别检查英文RST目录文件的接口列表是否与中文RST对应

- **混合输入**：分别按语言类型应用对应规则

#### 英文源文件查找

当输入为中文API文档时，需要查找对应的英文源文件：

| 英文源类型 | 后缀 | 说明 |
|-----------|------|------|
| Python代码 | `.py` | 包含docstrings的Python代码 |
| C++代码 | `.h`, `.cpp` | 包含 `///` 注释的C++代码 |
| YAML配置 | `.yaml`, `.yml` | MindSpore算子/接口定义文件（Tensor方法等） |
| 英文RST目录 | `.rst` | API索引/目录文件（如 `mindspore.nn.rst`） |

**典型路径映射：**

```text
中文RST(Python): docs/api/api_python/mindspore/nn/tanh.rst
英文Python代码: mindspore/python/mindspore/nn/tanh.py

中文RST(Python): docs/api/api_python/mindspore/Tensor/mindspore.Tensor.gather.rst
英文YAML文档: mindspore/ops/op_def/yaml/doc/gather_doc.yaml

中文RST(Python): docs/api/api_python/ops/mindspore.ops.func_strided_slice.rst
英文YAML文档: mindspore/ops/op_def/yaml/doc/strided_slice_doc.yaml

中文RST(Python): docs/api/api_python/mint/mindspore.mint.func_empty.rst
英文YAML文档: mindspore/ops/api_def/function_doc/empty_doc.yaml

中文RST(Python): docs/api/api_python/mindspore.nn.rst
英文RST目录: docs/api/api_python_en/mindspore.nn.rst

中文MD(C++): mindspore/docs/api/cpp_api/classmindspore_1_1Tensor.md
英文C++头文件: mindspore/core/include/mindspore/core/ops/tensor_impl.h
```

**YAML文档类型说明：**

| 类型 | 路径 | 用途 |
|------|------|------|
| function_doc | `mindspore/ops/op_def/yaml/doc/` | 函数接口文档 |
| method_doc | `mindspore/ops/op_def/yaml/doc/` | 方法接口文档 |
| function_doc(alternative) | `mindspore/ops/api_def/function_doc/` | 函数接口文档（备选） |
| method_doc(alternative) | `mindspore/ops/api_def/method_doc/` | 方法接口文档（备选） |

**查找策略：**

1. 根据模块路径推断
2. 根据API名称搜索YAML文档文件（优先搜索 `op_def/yaml/doc/` 目录）
3. 用户指定路径

#### API文档检查流程

1. **接收输入** → 解析输入类型和内容
2. **判断检查组合** → 根据内容特征选择通用+教程或通用+API
3. **执行格式检查** →

    - 中文RST文件：使用 `rules/api_python_zh_rules.md`
    - 中文MD文件：使用 `rules/api_cpp_zh_rules.md`
    - Python代码：使用 `rules/api_python_en_rules.md`（Python部分）
    - YAML文件：使用 `rules/api_python_en_rules.md`（YAML部分）
    - C++头文件：使用 `rules/api_cpp_en_rules.md`

4. **执行一致性检查** → 使用 `rules/api_python_consistency_rules.md` 检查中英文对应关系
5. **生成报告** → 按照"输出格式"生成Markdown报告，在"基本信息"中列出本次使用的规则文件
6. **保存文件** → 保存到本地并告知用户路径

## 输出格式

生成Markdown格式的检查报告，保存到本地文件。

### 报告结构

```markdown
# 文档质量检查报告

## 基本信息

| 项目 | 内容 |
|------|------|
| 检查时间 | [时间戳] |
| 输入类型 | [HTML页面/PR链接/本地文件] |
| 输入来源 | [具体URL或路径] |
| 检查组合 | [通用+教程/通用+API] |
| 使用规则 | [本次使用的规则文件列表，如 `api_python_zh_rules.md`、`api_python_en_rules.md`] |

## 检查结果概览

| 问题等级 | 数量 |
|----------|------|
| 严重问题 | X |
| 一般问题 | X |
| 建议优化 | X |
| **总计** | **X** |

## 详细问题列表

- 以下仅给出输出格式参考，编号以规则文件中列举的实际编号为准。
- 本章节的问题分类不需要给出具体的规则文件来源。

### 通用性问题

| 编号 | 问题标题 | 优先级 | 位置 | 问题描述 | 建议修复 |
|------|----------|--------|------|----------|----------|
| G-L-01 | 单词拼写问题 | 严重 | 第X行 | XX单词拼写错误 | XX单词应改为XX |
| G-U-01 | 使用被动语态 | 一般 | 第X行 | XXX这句话使用了被动语态 | 建议将这句话改为XXX |

### 教程相关问题（如适用）

| 编号 | 问题标题 | 优先级 | 位置 | 问题描述 | 建议修复 |
|------|----------|--------|------|----------|----------|
| T-C-04 | 运行结果为 | 建议 | 第X行 | 运行结果中出现WARNING信息 | 建议删除运行结果中的WARNING信息 |

### API文档问题（如适用）

| 编号 | 问题标题 | 优先级 | 位置 | 问题描述 | 建议修复 |
|------|----------|--------|------|----------|----------|
| I-PY-R-01 | 缺失Summary描述 | 严重 | 第X行 | 函数/类缺少描述接口功能的Summary | 添加Summary描述 |
| I-PY-A-01 | 参数类型错误 | 严重 | 第X行 | 参数类型格式不正确 | 使用 `arg (int): description` 格式 |
| I-CPP-F-01 | 文件命名错误 | 严重 | 文件名 | class文件命名不符合规范 | 使用 `class{namespace}_{classname}.md` 格式 |
| I-YA-P-01 | 位置参数缺少类型 | 一般 | 第X行 | 缺少参数类型说明 | 使用 `input (Tensor): 描述` 格式 |
| I-PY-EF-02 | 示例缺少续行符 | 一般 | 第X行 | 多行代码缺少 `...` 续行符 | 类/函数定义后使用 `...` 续行 |
| I-YA-M-01 | 公式格式错误 | 建议 | 第X行 | 行公式未缩进 | 使用 `.. math::` + 缩进格式 |

### 一致性问题（如适用）

| 编号 | 问题类型 | 优先级 | 位置 | 中文文档 | 英文源码 | 建议修复 |
|------|----------|--------|------|----------|----------|----------|
| I-CN-D-03 | 参数名不一致 | 严重 | 中文RST第X行 / Python代码第Y行 | `**input_data**` | `input_data` | 确保参数名完全一致 |
| I-CN-A-03 | 参数类型不一致 | 严重 | 中文RST第X行 / Python代码第Y行 | `(Tensor)` | `(int)` | 确保参数类型一致 |
| I-CN-S-01 | 描述语义差异 | 一般 | 中文RST第X行 / Python代码第Y行 | 中文描述... | 英文描述... | 保持描述语义一致 |
| I-CN-T-02 | 返回值类型不一致 | 一般 | 中文RST第X行 / Python代码第Y行 | 返回：Tensor | Returns: int | 确保返回值类型一致 |
| I-PR-M-01 | 中文新增无英文对应 | 建议 | PR #123 | 新增 `xxx.rst` 文档 | 需在对应文件中添加注释 |

## 改进建议

1. [🔴高] 建议...
2. [🟠中] 建议...
3. [🟢低] 建议...
```

### 输出文件

- 多输入或混合输入都合并到 `report.md`，如果该文件已存在，确保本次结果写入成功
- 报告内按输入源分章节（如 `## PR #1234`, `## 本地文件: xxx.rst`）
- 默认保存到当前目录，或用户指定路径

## 注意事项

- 对于PR链接，会分析PR中的文档变更部分
- 多个输入源合并到一个报告，按输入源分章节
- 检查过程会尽量获取页面完整内容
- 图片使用本地路径时无法验证，会标记为"需人工确认"
- 中文RST不包含示例，不需要检查样例部分