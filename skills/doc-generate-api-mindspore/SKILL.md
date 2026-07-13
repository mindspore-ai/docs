---
name: doc-generate-api-mindspore
description: Generate documentation for Python APIs, functions, classes, and modules — including English docstrings and Chinese RST docs. Use when adding docstrings to new functions or classes, writing API reference docs, creating examples, documenting classes, or following Python doc conventions. Triggers on phrases like "API documentation", "API docs", "document API", "write API documentation", "generate API docs", "API reference".
---

# API Documentation Guide (API文档生成指南)

This skill generates Python API documentation in **English docstrings** (`.py`) and/or **Chinese RST docs** (`.rst`).

## Workflow (工作流程)

### Before Generation (生成前)

0. **Ask generation scope**: Use the `question` tool to ask the user (single choice, labels in Chinese):
   - **两者都生成**（默认）— both English and Chinese
   - **仅英文** — English docstring only
   - **仅中文** — Chinese RST doc only

1. **Collect references**: PR links, test files, design docs, issue descriptions
2. **Read the source code**: Understand function signatures, parameter types, return values
3. **Verify parameter types**: Cross-reference type hints with references for untyped params
4. **Run or infer examples**: Get actual output values from test files or references
 5. **Check existing files** (if scope includes Chinese RST): Search the docs directory for existing `.rst` files.
    - Search method: (a) by dotted path pattern — use `glob` with `*<api_name>.rst` across `docs/`, or (b) by content — use `grep` for the API name within `.rst` files under `docs/`.
    - **Current API found** → Use its current filename; skip Naming Convention + Path Mapping + trimming prompt.
    - **Current API not found, but sibling APIs (other functions/classes in the same `.py` file) have RST files** → Derive the naming convention from the sibling's filename (e.g., if sibling uses `pkg.module.API`, trim the source filename segment); skip project-wide pattern alignment; proceed to step 6 Path Mapping only.
    - **No RST found for current API or any sibling** → Proceed to step 6 for full Naming Convention + Path Mapping.
6. **Identify naming + file paths** (only when no existing file): Determine the `.py` source path. Determine the dotted path via **Naming Convention** (filename = title = directive), check the depth threshold and ask the user if trimming is needed. Then determine the output **directory** via **Path Mapping**.

### Output Target (输出目标)

| Output | File Type | Location | Tools |
|--------|-----------|----------|-------|
| English docstring | `.py` | Directly into the Python source file | Read + Edit |
| Chinese RST doc | `.rst` | Project-specific (see Naming Convention + Path Mapping below) | Read + Edit / Write |

#### Naming Convention (命名规范)

**RST filename = title (1st line) = `.. py::` directive path.** One file per API, all three always identical.

The full dotted path from the package root is the default. For deeply nested paths, the user may trim intermediate levels — **all three** use the shorter path together.

Rules:
- **Default**: full path **starting from the package root** (e.g., `mindspore.ops.affine_grid`), always includes the package name. The package root is the top-level Python package directory in the repo (typically matches the repo name or main source dir), not inferred from internal import statements.
- **Trimming**: when full path is overly deep, user trims middle segments. Package + API name always kept.
- **Depth threshold with project pattern alignment**: Determine the dotted path as follows:
  1. Start from the full dotted path at the package root.
  2. Check existing `.rst` files in the project's docs directory. If the prevailing pattern consistently omits the source filename (e.g., `pkg.module.API` rather than `pkg.module.filename.API`), automatically trim accordingly to match.
  3. If the resulting path exceeds **4 levels** (e.g., `mindspore.a.b.c.ReLU` is 5 levels), **must ask** the user whether to trim further. Provide concrete trimming suggestions — list options that keep package + API name and remove different combinations of middle segments. 4 levels or fewer use the current path directly without asking.
- **Exception**: MindSpore ops `func_` prefix in filename, removed in title/directive.

The dotted path determines the filename, title, and directive — **all three must always be identical**.

Examples:
| File | Title | Directive |
|------|-------|-----------|
| `mindspore.nn.Tanh.rst` | `mindspore.nn.Tanh` | `.. py:class:: mindspore.nn.Tanh` |
| `mindspore.ops.AffineGrid.rst` | `mindspore.ops.AffineGrid` | `.. py:class:: mindspore.ops.AffineGrid` |
| `mindspore.ops.func_abs.rst`（例外）| `mindspore.ops.abs` | `.. py:function:: mindspore.ops.abs` |
| `mindspore.Tensor.abs.rst` | `mindspore.Tensor.abs` | `.. py:method:: mindspore.Tensor.abs` |

For the exception row (3rd), the `func_` prefix is present in the filename but omitted from the title and directive.

#### Path Mapping (路径映射)

Once the naming convention (dotted path) is determined, map it to the output directory (where the `.rst` file will be saved).

Examples (illustrative only, not an allowlist):

| Repository | Source | Dotted Path → Filename | Output Directory |
|------------|--------|----------------------|------------------|
| mindspore | `mindspore/python/mindspore/nn/tanh.py` | `mindspore.nn.Tanh` | `docs/api/api_python/nn/` |
| mindspore-lite | `mindspore-lite/python/api/model.py` | `mindspore_lite.model` | `docs/api/lite_api_python/` |
| lite_boost | `lite_boost/python/parallel/context_parallel.py` | `lite_boost.parallel.context_parallel` | `lite_boost/docs/api/lite_boost_api_python/lite_boost/` |

The directory is derived by: (a) checking existing docs dirs, (b) matching module hierarchy, (c) confirming `.rst` format from neighbors. The table above is only illustrative — **every repo follows this same process.**

If the directory still cannot be determined after applying these rules, **ask the user** where to save the `.rst` file.



### Generation (生成中)

1. **Load rules**: Read the corresponding rules file(s):
   - **仅英文** → `rules/python-docstring-guide.md`
   - **仅中文** → `rules/chinese-rst-guide.md`
   - **两者都生成** → Both `rules/python-docstring-guide.md` and `rules/chinese-rst-guide.md`

2. **Generate**: Apply the loaded rules to create or update the target file(s) at the mapped paths.

### Cross-check (交叉验证)

Compare English docstring and Chinese RST for consistency on shared content: params, return type, exception types, math formulas, opening description.

Do NOT flag: Chinese RST omits Examples and Supported Platforms, uses different heading formats.

- **两者都生成** → Fix inconsistencies directly
- **仅英文/仅中文** → If the other-language file exists, report discrepancies without modifying it. Skip if it does not exist.

### Quality Checklist (质量检查清单)

If any item is not satisfied, fix it directly.

#### English Docstring

- [ ] Summary in third person, includes purpose
- [ ] Args/Returns/Raises sections complete per signature
- [ ] Example is runnable and shows expected output
- [ ] Uses `r"""..."""` raw string prefix

#### Chinese RST

- [ ] RST filename = title = `.. py::` directive path, all three consistent (exception: `func_` prefix kept in filename, omitted in title/directive)
- [ ] Title followed by `=` underline before directive
- [ ] Correct heading: `参数：` / `返回：` / `异常：` / `输入：` / `输出：`
- [ ] Parameter format: `- **name** (Type) - Description.`
- [ ] No colons in description text
- [ ] No Examples or Supported Platforms sections
- [ ] Proper nouns kept in English (NumPy, MindSpore, etc.)

### After Generation (生成后)

- **仅英文** → Report the `.py` file path. If an existing Chinese RST was found with inconsistencies, list them.
- **仅中文** → Report the `.rst` file path. If an existing English docstring was found with inconsistencies, list them.
- **两者都生成** → Report both paths plus total APIs documented.

> **Tip**: After generation, additional references (PR links, test files, etc.) can be provided to refine accuracy.
