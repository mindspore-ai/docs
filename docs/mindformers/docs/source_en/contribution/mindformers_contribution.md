# MindSpore Transformers Contribution Guidelines

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/contribution/mindformers_contribution.md)

## Contributing Code to MindSpore Transformers

### Code Style Requirements

Please follow this style for MindSpore Transformers review, maintenance and development.

- Coding Guide

  The MindSpore Transformers community uses the `Python PEP 8` coding style. It is recommended to install the following plugins in your IDE to check code format: `Lizard`, `ShellCheck` and `PyLint`.

- Unit Testing Guide

  The MindSpore Transformers community uses the Python unit testing framework pytest. Annotation names need to reflect the design intent of the test case.

- Reconstruction Guide

  We encourage developers to reconstruct our code to eliminate code bad taste. All code should conform to coding style and testing style, and reconstructing code is no exception. The Lizard threshold for uncommented lines of code (nloc) is 100, and the circle complexity (cnc) threshold is 20. when a Lizard warning is received, the code to be merged must be reconstructed.

- Documentation Guide

  We use MarkdownLint to check Markdown document format. The following rules are modified based on the default configuration:

  1. MD007 (unordered list indent): the parameter indent is set to 4, indicating that all the contents of the unordered list need to be indented by 4 spaces.
  2. MD009 (space at the end of the line): the parameter br_spaces is set to 2, indicating that there can be either 0 or 2 spaces at the end of the line.
  3. MD029 (sequence number of ordered list): the parameter style is set to ordered, indicating ascending order.

### Fork-Pull Development Model Guide

- Fork MindSpore Transformers code repository

  Before submitting code to the MindSpore Transformers project, please make sure that you have forked this project to your own code repository. There may be parallel development between the MindSpore Transformers code repository and your own code repository, so please be aware of the consistency between them.

- Clone remote code repository

  If you want to download the code to your local computer, it is best to use the git method.

  ```shell
  # Clone repositories on Gitee
  git clone https://gitee.com/(insert_your_forked_repo)/mindformers.git
  ```

- Local Development Code

  `dev` is the development branch. Please pull the latest code from `dev` branch for development. And submit it to the `dev` branch when you submit your Pull Request.

  ```shell
  git checkout -b {new branch name} origin/dev
  ```

- Submit PR to MindSpore Transformers code repository

  In the last step, you need to pull a compare request between the new branch and the `MindSpore Transformers` master branch. After completing the pull request, `Jenkins CI` will be automatically set up for build testing. PR should be merged into the upstream dev branch as soon as possible to minimize the risk of merging.

  ```shell
  # Add all changes to the staging area
  git add

  # Check Update Status
  git status

  # To commit changes, add a commit header with the -m option
  git commit -m "The title of your commit"

  # Add a specific description of the commit, add a signature with the -s option, and modify the most recent commit with the -amend option.
  git commit -s --amend

  # Push changes to a new branch in the remote repository
  git push origin {New branch name}

  ```

### Documentation and Code Format

If you wish to merge custom models into the `MindSpore Transformers` code repository, there are a few things to keep in mind:

1. The file format and location should follow the norms.
2. Register the new model in the code to adapt it for higher-order interface use.

#### File Format and Location

1. The model code files are placed uniformly in the `research/{model_name}` folder in the following format.

    ```plaintext
    research/{model_name}
    ├── {model_name}
    | ├── {pretrain/finetune/predict}_{model_name}_{n}b.yaml
    | ├── convert_weight.py # Torch weights to MindSpore weights script (required for migration models)
    | ├── convert_reversed.py # MindSpore weights to Torch weights script (required for migration models)
    | ├── run_{model_name}.py # Running the code file
    | ├── {model_name}.py   # Model class code file
    | └── {model_name}_tokenizer.py # Tokenizer Code File
    ```

2. Model documentation is placed in the same `research/{model_name}` folder.

## Requirements for Submitting A PR

### Only One Commit

For multi-commit PRs, use the `squash` command to merge multiple commits into one. For example use:

```shell
git rebase -i HEAD~3
```

You can see:

```shell
pick 1234567 Add new function A
pick 89abcdef Fixed bugs in A
pick 01234567 Some optimizations to A
```

squash merge commit (can be simplified to abbreviations such as s, p, f, etc.)

```shell
pick 1234567 Add new function A
pick 89abcdef Fixed bugs in A
pick 01234567 Some optimizations to A
```

### PR Descriptions

Please use the following md template.

```markdown

### Related Issue

### Reason (purpose, problem solved, etc.)

### Description (what was done, what was changed)

### check list

#### Was a program review or root cause analysis of the problem completed (Y/N)

#### Whether UT/ST of functional modules was completed, executed and passed with results attached (Y/N)

#### Whether it involves modification of public components or external interfaces, and if so, the scope of modification and impact assessment should be given (Y/N)

#### Whether it involves the modification of information, and if so, the modification should be synchronized (Y/N)

```

### Access Control Requirements

1. Submitting a PR requires [signing a CLA](https://www.mindspore.cn/icla).

2. Submitting a PR requires passing the CI check, which needs to be manually restarted by commenting `/retest` under comments after the gate fails and the code is corrected.