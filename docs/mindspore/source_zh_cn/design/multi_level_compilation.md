# MindSpore 多级编译架构

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/design/multi_level_compilation.md)

MindSpore 2.3.0以前的版本，静态图模式采用整图下沉到Device侧执行机制，该机制在编译阶段耗时长、调试调优效率不高，
并且对已有的内存复用算法、执行序优化等功能支持不友好，因此MindSpore提出多级编译架构，旨在降低编译时长、提高静态图下调试调优效率，
得益于多级流水的设计，该架构下可实现静态Shape下性能持平或优于原整图下沉机制。

## 多级编译架构概述

## 多级编译O0级别介绍

## 多级编译O1级别介绍

## 多级编译O2级别介绍
