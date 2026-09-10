---
name: doc-release
description: MindSpore 文档发布新分支时，链接更替与组件清洗技能，用于清理组件和替换链接。
---

# OpenCode Skill: 文档发布 Skill

## 技能元信息

| 属性 | 值 |
|------|-----|
| 技能名称 | `文档发布 Skill` |
| 适用项目 | MindSpore 文档仓库 |
| 维护者 | MindSpore Docs Team |

---

## 技能描述

文档发布 Skill，用于 MindSpore 文档发布分支内容处理。发布新版本时，将文档从开发分支同步到发布分支，清理不发布的组件，并修正文档内部指向旧分支的链接。

本技能通过精确的正则表达式匹配，覆盖多种链接格式（OBS 资源链接、OBS Notebook 链接、atomgit 链接、官网链接、git clone 命令等），确保文档中的链接在新版本中正确指向目标分支。

**核心特性：**

1. `source_branch` 参数可配置为任意分支名（`master`、`r2.9.0`、`dev` 等）
2. 官网链接使用 `target_html_branch`，仓库链接使用 `tag`（仅 MindSpore 系列有 tag）
3. 增加链接有效性检测功能
4. 支持 `.ipynb` (Jupyter Notebook) 文件中的链接替换
5. 组件清洗，仅保留需要的组件目录

---

## 触发条件

当用户需要执行以下操作时触发本技能：

- 发布新版本，需要将文档从开发分支迁移到发布分支
- 清理文档仓库中不发布的组件文件夹
- 批量修正文档中指向旧分支的链接
- 验证替换后的仓库链接是否有效

---

## 限制条件

- 组件清洗仅在 `docs/docs` 目录下执行
- 链接替换作用于以下三个目录：
  - `docs/docs`
  - `docs/tutorials`
  - `docs/install`
- 链接替换支持以下文件类型：
  - `.md` (Markdown)
  - `.txt` (文本文件)
  - `.rst` (reStructuredText)
  - `.html` (HTML)
  - `.yml` / `.yaml` (YAML)
  - `.ipynb` (Jupyter Notebook)

---

## 输入参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `source_branch` | string | 否 | `master` | 源旧分支名称（可以是任意分支名） |
| `components_to_keep` | list[string] | 是 | - | 需要保留的组件白名单（`sample_code` 目录始终保留） |
| `component_branch_map` | map[string]object | 是 | - | 组件到目标分支的映射配置 |
| `enable_link_check` | boolean | 否 | `true` | 是否启用替换后的链接有效性检测 |

### component_branch_map 配置说明

每个组件配置包含以下字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `repo_name` | string | 是 | atomgit 仓库名称 |
| `has_git` | boolean | 是 | 是否有独立的 atomgit 仓库（`true`/`false`） |
| `target_html_branch` | string | 是 | 官网 HTML 分支（如 `r2.10.0`、`r2.0.0`、`r0.7`、`81RC1`） |
| `tag` | string | 否 | 仓库 tag（如 `v2.10.0`），**只有 MindSpore 有 tag** |
| `url_name` | string | 是 | 官网链接 `cn/` 后的路径名（如 `tutorials`、`docs`、`lite`、`mindquantum`） |

### 组件配置参考表

```python
# 只有 MindSpore 有 tag
# [repo_name, has_git, source_branch, target_html_branch, tag, url_name]
# ['tutorials', False, 'master', 'r2.10.0', '', 'tutorials'],
# ['mindscience', True, 'master', 'r0.7', '', 'mindscience'],
# ['mindflow', False, 'master', 'r0.3', '', 'mindflow'],
# ['mindearth', False, 'master', 'r0.3', '', 'mindearth'],
# ['mindformers', True, 'master', 'r2.0.0', '', 'mindformers'],
# ['mindspore', True, 'master', 'r2.10.0', 'v2.10.0', 'docs'],
# ['mindspore-lite', True, 'master', 'r2.10.0', 'r2.10', 'lite'],
# ['vllm-mindspore', True, 'master', 'r0.5.0', '', 'vllm_mindspore'],
# ['golden-stick', True, 'master', 'r1.3.0', '', 'golden_stick'],
# ['mindquantum', True, 'master', 'r0.11', '', 'mindquantum'],
# ['mindstudio', False, 'master', '81RC1', '', 'mindstudio'],
```

**字段说明：**

| 字段 | 说明 |
|------|------|
| `repo_name` | 仓库名 |
| `has_git` | 是否有独立仓库 |
| `source_branch` | 源分支（通常为 `master`） |
| `target_html_branch` | 目标 HTML 分支（用于官网链接和 OBS 链接） |
| `tag` | 仓库 tag（仅 MindSpore 有） |
| `url_name` | 官网链接 `cn/` 后的路径名 |

### 官网链接格式说明

官网链接形如：

```text
https://www.mindspore.cn/tutorials/zh-CN/master/index.html
                          ^^^^^^^^^         ^^^^^^
                          url_name          source_branch
```

其中 `url_name`（如 `tutorials`、`docs`、`lite`）是 `cn/` 后的路径名，`master` 是源分支。

**替换后：**

```text
https://www.mindspore.cn/tutorials/zh-CN/r2.10.0/index.html
```

### 特殊组件说明

| 组件 | 特殊处理 |
|------|----------|
| `tutorials` | 需额外处理教程专属链接：`mindspore.cn/tutorials/...` |
| `mindspore` | 需额外处理 docs 仓链接：`atomgit.com/mindspore/docs/...` 和 OBS 链接 |

### `.ipynb` 文件处理说明

`.ipynb` 是 JSON 格式文件，其中的链接通常以**未转义**的普通形式存储，例如：

```json
"source": [
  "[![View Source On AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/tutorials/source_en/dataset/record.ipynb)\n"
]
```

由于这些链接是普通字符串（非正则表达式），不需要额外的转义处理，直接将 `.ipynb` 加入文件后缀列表即可。

---

## 执行步骤

### 步骤 1：清理不发布的组件

在 `docs/docs` 目录下，删除不在白名单中的组件文件夹。`sample_code` 目录始终保留。

#### 执行逻辑

1. 构建保留列表：`["sample_code"] + components_to_keep`
2. 遍历 `docs/docs` 目录下的所有一级子目录
3. 如果目录名不在保留列表中，执行删除

#### 执行代码

```python
import os
import shutil

DOCS_COMPONENTS_DIR = "./docs/docs"

# 构建保留列表：sample_code + 用户指定的组件白名单
keep_list = ["sample_code"] + {{ parameters.components_to_keep }}

print(f"保留的组件: {keep_list}")

if os.path.isdir(DOCS_COMPONENTS_DIR):
    for entry in os.listdir(DOCS_COMPONENTS_DIR):
        full_path = os.path.join(DOCS_COMPONENTS_DIR, entry)
        if os.path.isdir(full_path) and entry not in keep_list:
            print(f"删除组件: {entry}")
            shutil.rmtree(full_path)
else:
    print(f"目录不存在: {DOCS_COMPONENTS_DIR}")
```

---

### 步骤 2：替换分支链接并检测有效性

在 `docs/docs`、`docs/tutorials`、`docs/install` 三个目录下，根据 `component_branch_map` 为每个组件替换对应的分支名。

#### 替换规则说明

脚本会为每个组件生成以下替换规则：

| 链接类型 | 使用的字段 | 示例（source_branch=master, target_html_branch=r2.10.0, tag=v2.10.0） |
|---------|-----------|----------------------------------------------------------------|
| OBS 资源链接 | `target_html_branch` | `website-images/master/resource/...` → `website-images/r2.10.0/resource/...` |
| OBS Notebook 链接 | `target_html_branch` | `notebook/master/tutorials/...` → `notebook/r2.10.0/tutorials/...` |
| 官网链接 | `target_html_branch` | `mindspore.cn/docs/zh-CN/master/` → `.../r2.10.0/` |
| atomgit docs 仓链接 | `target_html_branch` | `docs/blob/master/tutorials/...` → `docs/blob/r2.10.0/tutorials/...` |
| atomgit 仓库链接 | `tag`（如有） | `mindspore/blob/master/...` → `mindspore/blob/v2.10.0/...` |
| git clone 命令 | `tag`（如有） | `git clone -b master ...` → `git clone -b v2.10.0 ...` |

**注意：** 只有 MindSpore （`mindspore`）有 tag，其余用各自的branch。

#### 执行代码

```python
import os
import re
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ 链接检测函数 ============
def check_url(url, timeout=5):
    """检测 URL 是否可访问（HEAD 请求）"""
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        return resp.status_code < 400
    except Exception:
        return False

def validate_links(link_list, max_workers=5):
    """并发检测链接列表"""
    if not link_list:
        return

    logger.info(f"开始检测 {len(link_list)} 个链接...")
    failed_links = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(check_url, url): url for url in link_list}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                is_valid = future.result()
                if not is_valid:
                    failed_links.append(url)
                    logger.warning(f"⚠️ 链接可能无效: {url}")
            except Exception as e:
                failed_links.append(url)
                logger.error(f"检测失败 {url}: {e}")

    if failed_links:
        logger.error(f"发现 {len(failed_links)} 个可能无效的链接，请检查:")
        for link in failed_links:
            logger.error(f"  - {link}")
    else:
        logger.info("✅ 所有检测的链接均有效")

# ============ 主替换逻辑 ============
def replace_urls(target_paths, sub_list, enable_check=True):
    """遍历目标目录，执行链接替换"""
    all_replaced_links = []

    for tp in target_paths:
        if not os.path.exists(tp):
            logger.warning(f"路径不存在: {tp}")
            continue

        for root, dirs, files in os.walk(tp):
            for file in files:
                file_path = os.path.join(root, file)
                if not file.endswith(('.md', '.txt', '.rst', '.html', '.yml', '.yaml', '.ipynb')):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    new_content = content
                    for pattern, replacement in sub_list:
                        try:
                            new_content = re.sub(pattern, replacement, new_content, flags=re.DOTALL)
                        except re.error as e:
                            logger.error(f"正则表达式错误: {pattern}, {e}")
                            continue

                    if new_content != content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        logger.info(f"已更新: {file_path}")

                        if enable_check:
                            atomgit_links = re.findall(
                                r'https?://atomgit\.com/[^\s<>"\'\)]+',
                                new_content
                            )
                            all_replaced_links.extend(atomgit_links)

                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"处理失败 {file_path}: {e}")
                    continue

    if enable_check and all_replaced_links:
        validate_links(list(set(all_replaced_links)))

# ============ 配置区域 ============
target_paths = [
    './docs/docs',
    './docs/tutorials',
    './docs/install',
]

SOURCE_BRANCH = "{{ parameters.source_branch }}"
ENABLE_LINK_CHECK = {{ parameters.enable_link_check | default(True) | lower }}

# 组件配置：[repo_name, has_git, target_html_branch, tag, url_name]
# 只有 MindSpore 有 tag
component_configs = [
    {% for component, config in parameters.component_branch_map.items() %}
    [
        "{{ config.repo_name }}",
        {{ config.has_git | lower }},
        "{{ config.target_html_branch }}",
        "{{ config.tag if config.tag else '' }}",
        "{{ config.url_name }}"
    ],
    {% endfor %}
]

all_sub_list = []

# ============================================
# 第一部分：OBS 链接替换（通用）
# 使用 target_html_branch
# ============================================
for repo_name, has_git, target_html_branch, tag, url_name in component_configs:
    # OBS website-images 链接
    # 示例: website-images/master/resource/ -> website-images/r2.10.0/resource/
    all_sub_list.append([
        rf'(mindspore-website\.obs\.cn-north-4\.myhuaweicloud\.com/website-images)/{SOURCE_BRANCH}',
        rf'\1/{target_html_branch}'
    ])

    # OBS notebook 链接
    # 示例: notebook/master/tutorials/ -> notebook/r2.10.0/tutorials/
    all_sub_list.append([
        rf'(mindspore-website\.obs\.cn-north-4\.myhuaweicloud\.com/notebook)/{SOURCE_BRANCH}(/[\w\d/_.-]*?)',
        rf'\1/{target_html_branch}\2'
    ])

# ============================================
# 第二部分：各组件专属链接替换
# ============================================
for repo_name, has_git, target_html_branch, tag, url_name in component_configs:

    # ----- 处理 mindspore（docs 仓链接）-----
    if repo_name == 'mindspore':
        # atomgit docs 仓链接 - 使用 target_html_branch
        # 示例: docs/blob/master/tutorials/ -> docs/blob/r2.10.0/tutorials/
        all_sub_list.append([
            rf'(atomgit\.com/mindspore/docs/(?:blob|tree))/{SOURCE_BRANCH}(/[\w\d/_.-]*?)',
            rf'\1/{target_html_branch}\2'
        ])

    # ----- 处理 tutorials -----
    if repo_name == 'tutorials':
        # 官网教程链接 - 使用 target_html_branch
        # 示例: mindspore.cn/tutorials/zh-CN/master/ -> mindspore.cn/tutorials/zh-CN/r2.10.0/
        all_sub_list.append([
            rf'(mindspore\.cn/tutorials/[\w\d/_.-]*?)/{SOURCE_BRANCH}',
            rf'\1/{target_html_branch}'
        ])
        # 带 www 前缀
        all_sub_list.append([
            rf'(www\.mindspore\.cn/tutorials/[\w\d/_.-]*?)/{SOURCE_BRANCH}',
            rf'\1/{target_html_branch}'
        ])
        # atomgit docs 仓教程链接
        all_sub_list.append([
            rf'(atomgit\.com/mindspore/docs/(?:blob|tree))/{SOURCE_BRANCH}(/[\w\d/_.-]*?tutorials)',
            rf'\1/{target_html_branch}\2'
        ])

    # ----- 处理普通组件（非教程）-----
    if repo_name != 'tutorials':
        # 官网链接 - 使用 target_html_branch
        # 示例: mindspore.cn/docs/zh-CN/master/ -> mindspore.cn/docs/zh-CN/r2.10.0/
        all_sub_list.append([
            rf'(mindspore\.cn/{url_name}/[\w\d/_.-]*?)/{SOURCE_BRANCH}',
            rf'\1/{target_html_branch}'
        ])
        # 带 www 前缀
        all_sub_list.append([
            rf'(www\.mindspore\.cn/{url_name}/[\w\d/_.-]*?)/{SOURCE_BRANCH}',
            rf'\1/{target_html_branch}'
        ])

        # 仓库链接处理（仅当有独立仓库且有 tag 时）
        if has_git and tag:
            # atomgit 仓库链接 - 使用 tag
            # 示例: atomgit.com/mindspore/mindspore/blob/master/ -> blob/v2.10.0/
            all_sub_list.append([
                rf'(atomgit\.com/mindspore/{repo_name}/(?:blob|tree))/{SOURCE_BRANCH}(/[\w\d/_.-]*?)',
                rf'\1/{tag}\2'
            ])

            # git clone 命令（仅替换带 -b 参数的）
            # 示例: git clone -b master https://atomgit.com/mindspore/mindspore.git
            #    -> git clone -b v2.10.0 https://atomgit.com/mindspore/mindspore.git
            all_sub_list.append([
                f'git clone -b {SOURCE_BRANCH} https://atomgit.com/mindspore/{repo_name}.git',
                f'git clone -b {tag} https://atomgit.com/mindspore/{repo_name}.git'
            ])
        elif has_git and not tag:
            # 有独立仓库但没有 tag，只处理官网链接
            logger.info(f"⚠️ 组件 {repo_name} 有独立仓库但未配置 tag，仓库链接暂不替换")

        # 无独立仓库的组件（has_git=False），只处理官网链接

# ============================================
# 打印所有替换规则用于调试
# ============================================
logger.info("========== 替换规则列表 ==========")
logger.info(f"源分支: {SOURCE_BRANCH}")
logger.info(f"链接检测: {'启用' if ENABLE_LINK_CHECK else '禁用'}")
for pattern, replacement in all_sub_list:
    logger.info(f"替换: {pattern} -> {replacement}")
logger.info("==================================")

# ============================================
# 执行替换
# ============================================
replace_urls(target_paths, all_sub_list, enable_check=ENABLE_LINK_CHECK)
```

---

## 替换规则详解

### 1. OBS 链接替换（通用）

| 原始模式 | 替换后 | 使用的字段 |
|---------|--------|-----------|
| `website-images/{source_branch}/resource/` | `website-images/{target_html_branch}/resource/` | `target_html_branch` |
| `notebook/{source_branch}/tutorials/` | `notebook/{target_html_branch}/tutorials/` | `target_html_branch` |

### 2. 官网链接

| 原始模式 | 替换后 | 使用的字段 |
|---------|--------|-----------|
| `mindspore.cn/{url_name}/.../{source_branch}/` | `mindspore.cn/{url_name}/.../{target_html_branch}/` | `target_html_branch` |
| `www.mindspore.cn/{url_name}/.../{source_branch}/` | `www.mindspore.cn/{url_name}/.../{target_html_branch}/` | `target_html_branch` |

**示例（tutorials）：**

```text
https://www.mindspore.cn/tutorials/zh-CN/master/index.html
→ https://www.mindspore.cn/tutorials/zh-CN/r2.10.0/index.html
```

### 3. atomgit docs 仓链接

| 原始模式 | 替换后 | 使用的字段 |
|---------|--------|-----------|
| `docs/blob/{source_branch}/{path}` | `docs/blob/{target_html_branch}/{path}` | `target_html_branch` |
| `docs/tree/{source_branch}/{path}` | `docs/tree/{target_html_branch}/{path}` | `target_html_branch` |

### 4. atomgit 仓库链接（仅 MindSpore 系列）

| 原始模式 | 替换后 | 使用的字段 |
|---------|--------|-----------|
| `{repo}/(blob\|tree)/{source_branch}/{path}` | `{repo}/(blob\|tree)/{tag}/{path}` | `tag` |

### 5. git clone 命令（仅 MindSpore 系列）

| 原始模式 | 替换后 | 使用的字段 |
|---------|--------|-----------|
| `git clone {source_branch} https://...` | `git clone -b {tag} https://...` | `tag` |

---

## 使用示例

### 示例 1：发布 MindSpore 2.10.0 版本（完整组件列表）

```json
{
  "source_branch": "master",
  "components_to_keep": ["mindspore", "mindspore-lite", "mindformers", "mindquantum", "mindstudio"],
  "enable_link_check": true,
  "component_branch_map": {
    "tutorials": {
      "repo_name": "tutorials",
      "has_git": false,
      "target_html_branch": "r2.10.0",
      "tag": "",
      "url_name": "tutorials"
    },
    "mindspore": {
      "repo_name": "mindspore",
      "has_git": true,
      "target_html_branch": "r2.10.0",
      "tag": "v2.10.0",
      "url_name": "docs"
    },
    "mindspore-lite": {
      "repo_name": "mindspore-lite",
      "has_git": true,
      "target_html_branch": "r2.10.0",
      "tag": "r2.10",
      "url_name": "lite"
    },
    "mindformers": {
      "repo_name": "mindformers",
      "has_git": true,
      "target_html_branch": "r2.0.0",
      "tag": "",
      "url_name": "mindformers"
    },
    "mindquantum": {
      "repo_name": "mindquantum",
      "has_git": true,
      "target_html_branch": "r0.11",
      "tag": "",
      "url_name": "mindquantum"
    },
    "mindstudio": {
      "repo_name": "mindstudio",
      "has_git": false,
      "target_html_branch": "81RC1",
      "tag": "",
      "url_name": "mindstudio"
    }
  }
}
```

> **注意：** `components_to_keep` 中的每个组件都必须在 `component_branch_map` 中定义，否则该组件的链接不会被替换。

#### 执行效果

| 链接类型 | 原始 | 替换后 |
|---------|------|--------|
| 官网教程 | `mindspore.cn/tutorials/zh-CN/master/` | `mindspore.cn/tutorials/zh-CN/r2.10.0/` |
| 官网主文档 | `mindspore.cn/docs/zh-CN/master/` | `mindspore.cn/docs/zh-CN/r2.10.0/` |
| 官网 Lite | `mindspore.cn/lite/zh-CN/master/` | `mindspore.cn/lite/zh-CN/r2.10.0/` |
| 官网 MindFormers | `mindspore.cn/mindformers/zh-CN/master/` | `mindspore.cn/mindformers/zh-CN/r2.0.0/` |
| 官网 MindQuantum | `mindspore.cn/mindquantum/zh-CN/master/` | `mindspore.cn/mindquantum/zh-CN/r0.11/` |
| 官网 MindStudio | `mindspore.cn/mindstudio/zh-CN/master/` | `mindspore.cn/mindstudio/zh-CN/81RC1/` |
| atomgit docs 仓 | `docs/blob/master/tutorials/` | `docs/blob/r2.10.0/tutorials/` |
| atomgit mindspore 仓库 | `mindspore/blob/master/README` | `mindspore/blob/v2.10.0/README` |
| atomgit mindspore-lite 仓库 | `mindspore-lite/blob/master/README` | `mindspore-lite/blob/r2.10/README` |
| git clone (mindspore) | `git clone -b master .../mindspore.git` | `git clone -b v2.10.0 .../mindspore.git` |
| git clone (mindspore-lite) | `git clone -b master .../mindspore-lite.git` | `git clone -b r2.10 .../mindspore-lite.git` |
| git clone (mindformers) | `git clone -b master .../mindformers.git` | **暂不替换** |

---

## 字段使用总结

| 链接类型 | 使用的字段 | 说明 |
|---------|-----------|------|
| OBS 资源链接 | `target_html_branch` | 所有组件通用 |
| OBS Notebook 链接 | `target_html_branch` | 所有组件通用 |
| 官网链接（`mindspore.cn/{url_name}/...`） | `target_html_branch` | 使用各组件的 `url_name` |
| atomgit docs 仓链接 | `target_html_branch` | 仅 mindspore 组件 |
| atomgit 仓库链接 | `tag` | **仅 MindSpore 系列有 tag** |
| git clone 命令 | `tag` | **仅 MindSpore 系列有 tag** |

---

## 常见问题

### Q1: `url_name` 是什么？

`url_name` 是官网链接 `cn/` 后的路径名。例如：

```text
https://www.mindspore.cn/tutorials/zh-CN/master/index.html
                          ^^^^^^^^^
                          url_name = "tutorials"
```

常见值：`tutorials`、`docs`、`lite`、`mindquantum`、`mindformers`、`mindstudio` 等。

### Q2: `components_to_keep` 和 `component_branch_map` 是什么关系？

- `components_to_keep`：**保留哪些组件目录**（用于步骤 1 的组件清洗）
- `component_branch_map`：**每个组件的链接如何替换**（用于步骤 2 的链接替换）

**建议两者保持一致**：`components_to_keep` 中的每个组件都应该在 `component_branch_map` 中定义，否则该组件目录会被保留，但其链接不会被替换。

### Q3: 如何确定某个组件的 `target_html_branch`？

参考组件配置表：

```python
# ['组件名', has_git, 'master', 'target_html_branch', 'tag', 'url_name']
['tutorials', False, 'master', 'r2.10.0', '', 'tutorials'],
['mindformers', True, 'master', 'r2.0.0', '', 'mindformers'],
['mindspore', True, 'master', 'r2.10.0', 'v2.10.0', 'docs'],
['mindspore-lite', True, 'master', 'r2.10.0', 'r2.10', 'lite'],
['mindquantum', True, 'master', 'r0.11', '', 'mindquantum'],
['mindstudio', False, 'master', '81RC1', '', 'mindstudio'],
```

### Q4: 链接检测会误报吗？

可能。某些仓库可能设置了访问限制或需要登录。如果确认链接无误，可以设置 `"enable_link_check": false` 禁用检测。
