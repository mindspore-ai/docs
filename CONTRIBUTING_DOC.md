# Contributing Documents

You are welcome to contribute MindSpore documents. Documents that meet requirements will be displayed on the [MindSpore official website](https://www.mindspore.cn).

## Creating or Updating Documents

This project supports contribution documents in MarkDown and reStructuredText formats. You can create the ```.md``` or ```.rst``` files or modify existing documents.

## Submitting Modification

The procedure for submitting the modification is the same as that for submitting the code. For details, see [Code Contribution Guide](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md).

## Document Writing Specifications

- The title supports only the ATX style. The title and context must be separated by a blank line.

  ```text
  # Heading 1

  ## Heading 2

  ### Heading 3
  ```

- If the list title and content need to be displayed in different lines, add a blank line between the title and content. Otherwise, the line breaks may not be implemented.

  ```text
  - Title

    Content
  ```

- Anchors (hyperlinks) in the table of content can contain only Chinese characters, lowercase letters, and hyphens (-). Spaces or other special characters are not allowed. Otherwise, the link is invalid.

- Precautions are marked with a right angle bracket (>).

  ```text
  > Precautions
  ```

- References should be listed at the end of the document and marked in the document.

  ```text
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

  ```text
  """
  Comments on Python functions, methods, and classes
  """

  # Python code comments

  // C++ code comments

  ```

- A blank line must be added before and after an image and an image title. Otherwise, the typesetting will be abnormal.

  ```text
  Example:

  ![](./xxx.png)

  Figure 1: xxx

  The following content.
  ```
