# Multi-stream Concurrency

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/features/runtime/multistream_concurrency.md)

## Overview

During the training of large-scale deep learning models, the importance of communication and computation multi-stream concurrency for execution performance is self-evident in order to do as much communication and computation overlap as possible. To address this challenge, MindSpore implements automatic stream allocation and event insertion features to optimize the execution efficiency and resource utilization of the computational graph. The introduction of these features not only improves the concurrency of the computational graph, but also significantly reduces the device memory overhead, resulting in higher performance and lower latency in large model training.

## Basic Principle

Traditional multi-stream concurrency methods usually rely on manual configuration, which is not only cumbersome and error-prone, but also often difficult to achieve optimal concurrency when faced with complex computational graphs. MindSpore's automatic stream assignment feature automatically identifies and assigns concurrency opportunities in the computational graph by means of an intelligent algorithm, and assigns different operators to different streams for execution. This automated allocation process not only simplifies user operations, but also enables dynamic adjustment of stream allocation policies at runtime to accommodate different computing environments and resource conditions.

![multi_stream](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_en/design/images/multi_level_compilation/jit_level_multi_stream.png)

The principles are as follows:

1. Identify communication operators and computation operators based on the execution order.
2. Automatic multi-stream allocation of computational communication operators to achieve concurrent execution.
3. Data dependency between multiple streams is achieved by inserting events at the boundary locations of multiple streams for non-blocking downstreaming of hosts.

**Multi-stream Management**

In order to achieve the above effect of concurrent execution of multiple streams, multi-stream management is an important technique aimed at efficiently managing and scheduling the streams (Streams) on the computing devices to optimize the execution efficiency and resource utilization of the computational graph. Device multi-stream management ensures efficient concurrent execution of computing and communication tasks in a multi-computing resource environment through intelligent stream allocation and scheduling policies, thus improving overall performance.

![stream_manager](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/features/runtime/images/stream_manager.png)

**Stream Manager** plays a central role. It is responsible for the creation, distribution and destruction of streams, ensuring that each computational task is executed on the appropriate stream. The stream manager schedules tasks to different streams based on the type and priority of the task and the load on the device to achieve optimal resource utilization and task concurrency.

**Event Manager** monitors and manages synchronization and dependencies between streams. By logging and triggering events, the event manager ensures that tasks on different streams are executed in the correct order, avoiding data contention and resource conflicts. The event manager also supports the triggering and processing of asynchronous events (e.g., memory recalls), which further enhances the concurrency and responsiveness of the system.
