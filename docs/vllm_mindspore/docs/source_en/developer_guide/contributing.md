# Contribution Guidelines

[![View Source](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/developer_guide/contributing.md)

## Contributor License Agreement

Before submitting code to the MindSpore community, you need to sign the Contributor License Agreement (CLA). Individual contributors should refer to the [ICLA Online Document](https://www.mindspore.cn/icla).

## Quick Start

- Fork the repository on [Gitee](https://gitee.com/mindspore/vllm-mindspore).
- Refer to [README.md](https://gitee.com/mindspore/vllm-mindspore/blob/master/README.md) and the installation page for project information and build instructions.

## Supporting New Models

To support a new model for vLLM MindSpore code repository, please note the following:

- **Follow file format and location specifications.** Model code files should be placed under the `vllm_mindspore/model_executor` directory, organized in corresponding subfolders by model type.
- **Implement models using MindSpore interfaces with jit static graph support.** Model definitions in vLLM MindSpore must be implemented using MindSpore interfaces. Since MindSpore's static graph mode offers performance advantages, models should support execution via @jit static graphs. For reference, see the [Qwen2.5](https://gitee.com/mindspore/vllm-mindspore/blob/master/vllm_mindspore/model_executor/models/qwen2.py) implementation.
- **Register new models in vLLM MindSpore.** After implementing the model structure, register it in vLLM MindSpore by adding it to `_NATIVE_MODELS` in `vllm_mindspore/model_executor/models/registry.py`.
- **Write unit tests.** New models must include corresponding unit tests. Refer to the [Qwen2.5 testcases](https://gitee.com/mindspore/vllm-mindspore/blob/master/tests/st/python/cases_parallel/vllm_qwen_7b.py) for examples.

## Contribution Process

### Code Style

Follow these guidelines for community code review, maintenance, and development.

- **Coding Standards:** Use vLLM community code checking tools: yapf, codespell, ruff, isort, and mypy. For more details, see the [Toolchain Usage Guide](https://gitee.com/mindspore/vllm-mindspore/blob/master/codecheck_toolkits/README.md).
- **Unit Testing Guidelines:** vLLM MindSpore uses the [pytest](http://www.pytest.org/en/latest/) framework. Test names should clearly reflect their purpose.
- **Refactoring Guidelines:** Developers are encouraged to refactor code to eliminate [code smells](https://en.wikipedia.org/wiki/Code_smell). All code, including refactored code, must adhere to coding and testing standards.

### Fork-Pull Development Model

- **Fork the vLLM MindSpore Repository:** Before submitting code, fork the project to your own repository. Ensure consistency between the vLLM MindSpore repository and your fork during parallel development.

- **Clone the Remote Repository:** users can use git to pull the source code:

  ```shell
  # On Gitee:
  git clone https://gitee.com/{insert_your_forked_repo}/vllm-mindspore.git
  git remote add upstream https://gitee.com/mindspore/vllm-mindspore.git
  ```

- **Local Development:** To avoid branch inconsistencies, switch to a new branch:

  ```shell
  git checkout -b {new_branch_name} origin/master
  ```

  For version branches or downstream development, fix upstream bugs before modifying code.
- **Push Changes to Remote Repository:** After updating the code, push changes:

  ```shell
  git add .
  git status # Check update status.
  git commit -m "Your commit title"
  git commit -s --amend # Add detailed commit description.
  git push origin {new_branch_name}
  ```

- **Create a Pull Request to vLLM MindSpore:** Compare and create a PR between your branch and the vLLM MindSpore master branch. After submission, manually trigger CI checks with `/retest` in the comments. PRs should be merged into upstream master promptly to minimize merge risks.

### Reporting Issues

To contribute by reporting issues, follow these guidelines:

- Specify your environment versions (vLLM MindSpore, MindFormers, MindSpore, OS, Python, etc.).
- Indicate whether it's a bug report or feature request.
- Label the issue type for visibility on the issue board.
- Describe the problem and expected resolution.
- Provide detailed reproduction steps.
- Add special notes for reviewers.

**Issue Notes:**

- **Comment first when processing an issue,** inform others that you would start to fix this issue.
- **For long-unresolved issues**, verify the problem before attempting a fix.
- **If you resolve your own reported issue**, notify others before closing it.

### Submitting PRs

- For major new features, include a design proposal.
- After consensus via issue discussion and design review, develop in your fork and submit a PR.
- Each PR requires at least two LGTM labels from reviewers (excluding the PR author).
- After thorough discussion, the PR will be merged, abandoned, or rejected based on the outcome.

**PR Notes:**

- Avoid unrelated changes.
- Maintain clean commit history.
- Keep your branch synchronized with master.
- For bug-fix PRs, ensure all related issues are referenced.

Thank you for your interest in contributing to vLLM MindSpore. We welcome and value all forms of collaboration.
