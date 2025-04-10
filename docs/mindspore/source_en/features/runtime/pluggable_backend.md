# Multi-backend Access

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/features/runtime/pluggable_backend.md)

## Overview

In order to meet the rapid docking requirements of new backend and new hardware, MindSpore supports plug-in, low-cost and rapid docking of third-party backend on the basis of MindIR through an open architecture. Third-party backend does not need to pay attention to the data structure and implementation of the current existing backend, and only needs to use MindIR as an input to realize its own backend and functionality, which will be loaded with independent so registration. The functionality of different backends will be isolated from each other.

## Interface

Multi-backend implementation: for the specified backend to use via mindspore.jit(backend="xx"), see [jit interface](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.jit.html#mindspore.jit).

## Basic Principle

![multi_backend](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/features/runtime/images/multi_backend.png)

The MindSpore multi-backend docking schematic is shown above, with the core idea being:

1. The backend management module provides C++ external interfaces (Build and Run for backendmanager) and internal interfaces (Build and Run for the base class backend).
2. backendmanager external interface, mainly provided to the front-end MindIR docking back-end functionality for front-end and back-end decoupling.
3. Base class backend internal interface, mainly provided to the respective backend to achieve Build and Run functions.
4. Each back-end function is an independent so for the back-end management module to dynamically load scheduling.

After understanding the core idea of MindSpore's multi-backend docking, the main tasks when adding a new backend are as follows:

1. mindspore.jit(backend="xx") interface adds new backend type.
2. The new backend class inherits from the base class backend and implements the corresponding Build and Run functions.
3. The new backend code is compiled into a separate so and registered to the backend management module.