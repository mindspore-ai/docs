# LiteBoost Introduction

## Overview

LiteBoost is an inference acceleration toolkit for Ascend hardware, built on top of MindSpore Lite. It provides high-performance custom operators, multi-card parallel inference, quantization and sparsity, and other inference acceleration capabilities. LiteBoost builds on the PyTorch interface and deeply invokes Ascend CANN `aclnn` interfaces through C++ custom operators. It combines optimized Attention and RoPE implementations at the Python layer with HCCL multi-card communication to achieve end-to-end inference acceleration.

## Core Capabilities

### High-Performance Custom Operators

- Provides an easy-to-use interface by integrating CANN fused operators, enabling quick adoption of fused operators to improve model inference performance.
- Supports developing custom fused operators and exposing them through LiteBoost interfaces to improve model inference performance with PyTorch.

### Multi-Card Parallelism

- Supports multiple parallel strategies such as TP, CP, SP, and DP.
- Adapts and optimizes for different algorithm models through different parallel strategies, provides a simple and easy-to-use experience for open-source models, and improves developers’ ability to enable multi-card parallelism.

## Technical Architecture

LiteBoost adopts a dual-layer architecture of **C++ Operator Layer + Python Acceleration Layer**:

- **C++ Operator Layer**: Registers custom operators through the PyTorch `TORCH_LIBRARY` mechanism, compiles them into shared libraries, and deeply invokes Ascend CANN `aclnn` interfaces to fully leverage Ascend NPU hardware performance, and will continue to develop custom operators to improve the inference performance of this component.
- **Python Acceleration Layer**: Encapsulates Python bindings for C++ operators, optimized Attention Layers, and the HCCL-based multi-card parallel solution, providing a clean and easy-to-use Python API, and will continue to be updated and add acceleration optimizations related to quantization and sparsity.
