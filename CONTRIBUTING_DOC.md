# Contributing Documents

[查看中文](./CONTRIBUTING_DOC_CN.md)

You are welcome to contribute MindSpore documents. Documents that meet requirements will be displayed on the [MindSpore official website](https://www.mindspore.cn).

## Creating or Updating Documents

This project supports contribution documents in MarkDown and reStructuredText formats. You can create the ```.md``` or ```.rst``` files or modify existing documents.

## Submitting Modification

The procedure for submitting the modification is the same as that for submitting the code. For details, see [Code Contribution Guide](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md).

## Document Writing Specifications

- The title supports only the ATX style. The title and context must be separated by a blank line.

  ```markdown
  # Heading 1

  ## Heading 2

  ### Heading 3
  ```

- If the list title and content need to be displayed in different lines, add a blank line between the title and content. Otherwise, the line breaks may not be implemented.

  ```markdown
  - Title

    Content
  ```

- Anchors (hyperlinks) in the table of content can contain only Chinese characters, lowercase letters, and hyphens (-). Spaces or other special characters are not allowed. Otherwise, the link is invalid.

- Precautions are marked with a right angle bracket (>).

  ```markdown
  > Precautions
  ```

- References should be listed at the end of the document and marked in the document.

  ```markdown
  Add a [number] after the referenced text or image description.

  ## References

  [1] Author. [Document Name](http://xxx).

  [2] Author. Document Name.
  ```

- Comments in the sample code must comply with the following requirements:

    - Comments are written in English.
    - Use ```"""``` to comment out Python functions, methods, and classes.
    - Use ```#``` to comment out other Python code.
    - Use ```//``` to comment out C++ code.

  ```markdown
  """
  Comments on Python functions, methods, and classes
  """

  # Python code comments

  // C++ code comments

  ```

- A blank line must be added before and after an image and an image title. Otherwise, the typesetting will be abnormal. For example as correctly:

   ```markdown
  Example:

  ![](./xxx.png)

  Figure 1: xxx

  The following content.
  ```

- A blank line must be added before and after a table. Otherwise, the typesetting will be abnormal. Tables are not supported in ordered or unordered lists. For example as correctly:

  ```markdown
  ## Title

  | Header1  | Header2
  | :-----   | :----
  | Body I1  | Body I2
  | Body II1 | Body II2

  The following content.
  ```

- Mark the reference interface, path name, file name in the tutorial and document with "\` \`". If it's a function or method, don't use parentheses at the end. For example:

    - Reference method

    ```markdown
    Use the `map` method.
    ```

    - Reference code

    ```markdown
    `batch_size`: number of data in each group.
    ```

    - Reference path

    ```markdown
    Decompress the dataset and store it in `./MNIST_Data`.
    ```

    - Reference file name

    ```markdown
    Other dependencies is described in `requirements.txt`.
    ```

- In tutorials and documents, the contents that need to be replaced need additional annotation. In the body, a "*" should be added before and after the content. In the code snippet, the content should be annotated with "{}". For example:

    - In body

    ```markdown
    Need to replace your local path *your_ path*.
    ```

    - In code snippet

    ```markdown
    conda activate {your_env_name}
    ```

## Markdown Check

Markdownlint is a tool for checking the correctness of the Markdown file format. It can perform a comprehensive check on the Markdown file according to the set rules and the new rules created by users.

Among them, MindSpore CI modified the following rules based on the default configuration:

MD007(Unordered list indentation) sets the `indent` as 4; MD009 (Trailing spaces) sets the `br_spaces` as 2; MD029 (Ordered list item prefix) sets the `style` as ordered.

For other detailed rules, please refer to [RULES](https://github.com/markdownlint/markdownlint/blob/master/docs/RULES.md).
