# Differences in Different Platforms

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/platform_differences.md)

## Overview

In distributed training, different hardware platforms (Ascend, CPU or GPU) support different characters, and users can choose the corresponding distributed startup method, parallel mode and optimization method according to their platforms.

### Differences in Startup Methods

- Ascend supports dynamic cluster, mpirun, and rank table startup.
- GPU supports dynamic cluster and mpirun startup.
- CPU only supports dynamic cluster startup.

For the detailed process, refer to [startup methods](https://www.mindspore.cn/tutorials/experts/en/master/parallel/startup_method.html).

### Differences in Parallel Methods

- Ascend and GPUs support all methods of parallel, including data parallel, semi-automatic parallel, automatic parallel, and more.
- CPU only supports data parallel.

For the detailed process, refer to [data parallel](https://www.mindspore.cn/tutorials/experts/en/master/parallel/data_parallel.html), [semi-automatic parallel](https://www.mindspore.cn/tutorials/experts/en/master/parallel/semi_auto_parallel.html), [auto-parallel](https://www.mindspore.cn/tutorials/experts/en/master/parallel/auto_parallel.html).

### Differences in Optimization Feature Support

- Ascend supports all optimization features.
- GPU support optimization features other than communication subgraph extraction and multiplexing.
- CPU does not support optimization features.

For the detailed process, refer to [optimization methods](https://www.mindspore.cn/tutorials/experts/en/master/parallel/optimize_technique.html).
