# Tools

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/faq/tools.md)

## Q: When using the overflow detection Dump feature, I encounter the error `RuntimeError: aclnnAllFiniteGetWorkspaceSize call failed, please check!`. How can I resolve this?

A: This error typically occurs because the custom operators used by the overflow detection feature are incompatible with the current CANN version. The overflow detection Dump functionality in MindSpore has strict version requirements for CANN. A higher-version MindSpore is not compatible with a lower-version CANN.

To resolve this issue, consider the following approaches:

- Upgrade your CANN version

    Please refer to the official [Version Compatibility Matrix](https://www.mindspore.cn/install/en/) to install a CANN version that matches or exceeds the requirement for your MindSpore version.

- Switch to statistic-based Dump or other Dump methods

  If upgrading CANN is not immediately feasible, you can disable overflow detection Dump and instead use:
  - Statistic Dump: Records tensor statistics such as maximum and minimum values to help identify potential overflow;
  - Full Dump or Selective Dump: Saves intermediate tensor data for offline analysis of numerical anomalies, detailed please refer to [Using Dump in the Graph Mode](https://www.mindspore.cn/tutorials/en/master/debug/dump.html).

These workarounds can effectively avoid the runtime error while still enabling you to investigate overflow issues.
