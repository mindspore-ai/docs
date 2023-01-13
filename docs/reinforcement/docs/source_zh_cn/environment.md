# 强化学习环境接入

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/reinforcement/docs/source_zh_cn/environment.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

强化学习领域中，智能体与环境交互过程中，学习策略来使得数值化的收益信号最大化。“环境”作为待解决的问题，是强化学习领域中重要的要素。

目前强化学习使用的环境种类繁多：[Mujoco](https://github.com/deepmind/mujoco)、[MPE](https://github.com/openai/multiagent-particle-envs)、[Atari](https://github.com/gsurma/atari)、[PySC2](https://www.github.com/deepmind/pysc2)、[SMAC](https://github/oxwhirl/smac)、[TORCS](https://github.com/ugo-nama-kun/gym_torcs)、[Isaac](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)等，目前MindSpore Reinforcement接入了Gym、SMAC两个环境，后续随着算法的丰富，还会逐渐接入更多的环境。本文将介绍如何在MindSpore Reinforcement下接入第三方环境。

## 将环境Python函数封装为算子

在此之前，先介绍一下静态图和动态图模式。

- 动态图模式下，程序按照代码的编写顺序逐行执行，编译器将神经网络中的各个算子逐一下发到设备进行计算操作，方便用户编写和调试神经网络模型。

- 静态图模式下，程序在编译执行时，会将开发者定义的算法编译成一张计算图。在这个过程中，编译器可以通过使用图优化技术来降低资源开销，获得更好的执行性能。

由于静态图模式支持的语法是Python语言的子集，而常用的环境一般使用Python接口实现交互，二者之间的语法差异往往会造成图编译错误。对于这个问题，开发者可以使用`PyFunc`算子将Python函数封装为一个MindSpore计算图中的算子。

接下来以gym为例，将`env.reset()`封装为一个MindSpore计算图中的算子：

下面的代码中创建了一个`CartPole-v0`的环境，执行`env.reset()`方法，可以看到`state`的类型是`numpy.ndarray`，数据类型和维度分别是`np.float64`和`(4,)`。

```python
import gym

env = gym.make('CartPole-v0')
state = env.reset()
print('type: {}, shape: {}, dtype: {}'.format(type(state), state.dtype, state.shape))

# Result:
#   type: <class 'numpy.ndarray'>, shape: (4,), dtype: float64
```

接下来，使用`PyFunc`算子将`env.reset()`封装为一个MindSpore算子：

- `fn`指定需要封装的Python函数名，既可以是普通的函数，也可以是成员函数。
- `in_types`和`in_shapes`指定输入的数据类型和维度。`env.reset`没有入参，因此填写空的列表。
- `out_types`，`out_shapes`指定返回值的数据类型和维度。从之前的执行结果可以看到，`env.reset()`返回值是一个numpy数组，数据类型和维度分别是`np.float64`和`(4,)`，因此填写`[ms.float64,]`和`[(4,),]`。
- `PyFunc`返回值是个tuple(Tensor)。
- 更加详细的使用说明[参考](https://gitee.com/mindspore/mindspore/blob/r2.0.0-alpha/mindspore/python/mindspore/ops/operations/other_ops.py)。

## 环境和算法解耦

强化学习算法通常应该具备良好的泛化性，例如解决`HalfCheetah`的算法也应该能够解决`Pendulum`。为了贯彻泛化性的要求，有必要将环境和算法其余部分进行解耦，从而确保在更换环境后，脚本中的其余部分尽量少的修改。建议开发者参考`Environment`对环境进行封装。

```python
class Environment(nn.Cell):
    def __init__(self):
        super(Environment, self).__init__(auto_prefix=False)

    def reset(self):
        pass

    def step(self, action):
        pass

    @property
    def action_space(self) -> Space:
        pass

    @property
    def observation_space(self) -> Space:
        pass

    @property
    def reward_space(self) -> Space:
        pass

    @property
    def done_space(self) -> Space:
        pass
```

`Environment`除了提供`reset`和`step`等与环境交互的接口之外，还需要提供`action_space`、`observation_space`等方法，这些接口返回[Space](https://mindspore.cn/reinforcement/docs/zh-CN/r0.6.0-alpha/reinforcement.html#mindspore_rl.environment.Space)类型。算法可以根据`Space`信息：

- 获取环境的状态空间和动作空间的维度，用于构建神经网络。
- 读取合法的动作范围，对策略网络给出的动作进行缩放和裁剪。
- 识别环境的动作空间是离散的还是连续的，选择采用连续分布还是离散分布对环境探索。
