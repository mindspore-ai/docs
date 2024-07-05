# API注册机制介绍

## API组件注册机制

*介绍MIndFormers中各种API的注册机制以及使用示例，开发者可以通过该方式将模块进行注册，方便其他用户使用高阶接口进行调用，可注册模块类型包括'trainer'、'pipeline'、'dataset'等*

## API组件Build机制

*介绍MIndFormers中的build功能，并提供了相关案例，通过build接口可以将经过注册的模块进行实例化，对应模块使用build_trainer、build_pipeline、build_dataset等*