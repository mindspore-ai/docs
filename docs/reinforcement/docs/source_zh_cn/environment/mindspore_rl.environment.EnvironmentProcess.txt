.. py:class:: mindspore_rl.environment.EnvironmentProcess(proc_no, env_num, envs, actions, observations, initial_states)

    负责创建一个独立进程用作与一个或多个环境交互。

    参数：
        - **proc_no** (int) - 被分配的进程号。
        - **env_num** (int) - 传入此进程的环境数量。
        - **envs** (list(Environment)) - 包含环境实例（继承Environment类）的List。
        - **actions** (Queue) - 用于将动作传递给环境进程的队列。
        - **observations** (Queue) - 用于将状态传递给环境进程的队列。
        - **initial_states** (Queue) - 用于将初始状态传递给环境进程的队列。

    .. py:method:: run()

        在子进程中运行的方法，可以在子类中重写。