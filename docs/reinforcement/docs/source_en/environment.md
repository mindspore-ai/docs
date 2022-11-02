# Reinforcement Learning Environment Access

<a href="https://gitee.com/mindspore/docs/blob/master/docs/reinforcement/docs/source_en/environment.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

In the field of reinforcement learning, learning strategy maximizes numerical gain signals during interaction between an intelligent body and its environment. The "environment" is an important element in the field of reinforcement learning as a problem to be solved.

A wide variety of environments are currently used for reinforcement learning: [Mujoco](https://github.com/deepmind/mujoco), [MPE](https://github.com/openai/multiagent-particle-envs), [Atari]( https://github.com/gsurma/atari), [PySC2](https://www.github.com/deepmind/pysc2), [SMAC](https://github/oxwhirl/smac), [TORCS](https: //github.com/ugo-nama-kun/gym_torcs), [Isaac](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs), etc. Currently MindSpore Reinforcement is connected to two environments Gym and SMAC, and will gradually access more environments with the subsequent enrichment of algorithms. In this article, we will introduce how to access the third-party environment under MindSpore Reinforcement.

## Encapsulating Environmental Python Functions as Operators

Before that, introduce the static and dynamic graph modes.

- In dynamic graph mode, the program is executed line by line in the order in which the code is written, and the compiler sends down the individual operators in the neural network to the device one by one for computation operations, making it easy for the user to write and debug the neural network model.

- In static graph mode, the program compiles the developer-defined algorithm into a computation graph when the program is compiled for execution. In the process, the compiler can reduce resource overhead to obtain better execution performance by using graph optimization techniques.

Since the syntax supported by the static graph mode is a subset of the Python language, and commonly-used environments generally use the Python interface to implement interactions. The syntax differences between the two often result in graph compilation errors. For this problem, developers can use the `PyFunc` operator to encapsulate a Python function as an operator in a MindSpore computation graph.

Next, using gym as an example, encapsulate `env.reset()` as an operator in a MindSpore computation graph.

The following code creates a `CartPole-v0` environment and executes the `env.reset()` method. You can see that the type of `state` is `numpy.ndarray`, and the data type and dimension are `np.float64` and `(4,)` respectively.

```python
import gym

env = gym.make('CartPole-v0')
state = env.reset()
print('type: {}, shape: {}, dtype: {}'.format(type(state), state.dtype, state.shape))

# Result:
#   type: <class 'numpy.ndarray'>, shape: (4,), dtype: float64
```

`env.reset()` is encapsulated into a MindSpore operator by using the `PyFunc` operator.

- `fn` specifies the name of the Python function to be encapsulated, either as a normal function or as a member function.
- `in_types` and `in_shapes` specify the input data types and dimensions. `env.reset` has no input, so it fills in an empty list.
- `out_types`, `out_shapes` specify the data types and dimensions of the returned values. From the previous execution, it can be seen that `env.reset()` returns a numpy array with data type and dimension `np.float64` and `(4,)` respectively, so `[ms.float64,]` and `[(4,),]` are filled in.
- `PyFunc` returns tuple(Tensor).
- For more detailed instructions, refer to the [reference](https://gitee.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/ops/operations/other_ops.py).

## Decoupling Environment and Algorithms

Reinforcement learning algorithms should usually have good generalization, e.g., an algorithm that solves `HalfCheetah` should also be able to solve `Pendulum`. In order to implement the generalization, it is necessary to decouple the environment from the rest of the algorithm, thus ensuring that the rest of the script is modified as little as possible after changing the environment. It is recommended that developers refer to `Environment` to encapsulate the environment.

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

`Environment` needs to provide methods such as `action_space` and `observation_space`, in addition to interfaces for interacting with the environment, such as `reset` and `step`, which return [Space](https://mindspore.cn/reinforcement/docs/en/master/reinforcement.html#mindspore_rl.environment.Space) type. The algorithm can achieve the following operations based on the `Space` information:

- obtain the dimensions of the state space and action space in the environment, which used to construct the neural network.
- read the range of legal actions, and scale and crop the actions given by the policy network.
- Identify whether the action space of the environment is discrete or continuous, and choose whether to explore the environment by using a continuous or discrete distribution.
