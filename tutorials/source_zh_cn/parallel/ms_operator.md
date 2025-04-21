# 在K8S集群上进行分布式训练

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_zh_cn/parallel/ms_operator.md)

MindSpore Operator是遵循Kubernetes的Operator模式（基于CRD-Custom Resource Definition功能），实现的在Kubernetes上进行分布式训练的插件。其中，MindSpore Operator在CRD中定义了Scheduler、PS、Worker三种角色，用户只需通过简单的YAML文件配置，就可以轻松地在K8S上进行分布式训练。MindSpore Operator的代码仓详见：[ms-operator](https://gitee.com/mindspore/ms-operator/)。

## 安装

安装方法可以有以下三种：

1. 使用YAML直接安装

    ```shell
    kubectl apply -f deploy/v1/ms-operator.yaml
    ```

    安装后：

    使用`kubectl get pods --all-namespaces`，即可看到namespace为ms-operator-system的部署任务。

    使用`kubectl describe pod ms-operator-controller-manager-xxx-xxx -n ms-operator-system`，可查看pod的详细信息。

2. 使用make deploy安装

    ```shell
    make deploy IMG=swr.cn-south-1.myhuaweicloud.com/mindspore/ms-operator:latest
    ```

3. 本地调试环境

    ```shell
    make run
    ```

## 样例

当前ms-operator支持普通单Worker训练、PS模式的单Worker训练以及自动并行（例如数据并行、模型并行等）的Scheduler、Worker启动。

在[config/samples/](https://gitee.com/mindspore/ms-operator/tree/master/config/samples)中有运行样例。以数据并行的Scheduler、Worker启动为例，其中数据集和网络脚本需提前准备：

```shell
kubectl apply -f config/samples/ms_wide_deep_dataparallel.yaml
```

使用`kubectl get all -o wide`即可看到集群中启动的Scheduler和Worker，以及Scheduler对应的Service。

## 开发指南

### 核心代码

`pkg/apis/v1/msjob_types.go`中为MSJob的CRD定义。

`pkg/controllers/v1/msjob_controller.go`中为MSJob controller的核心逻辑。

### 镜像制作、上传

如需修改ms-operator代码并制作上传镜像，可参考以下命令：

```shell
make docker-build IMG={image_name}:{tag}
docker push {image_name}:{tag}
```

### YAML文件配置说明

以自研组网的数据并行为例，介绍MSJob的YAML配置，如`runPolicy`、`successPolicy`、各个角色数量、mindspore镜像、文件挂载等，用户需根据自己的实际需要进行配置，YAML文件配置项中不要使用下划线。

```yaml
apiVersion: mindspore.gitee.com/v1
kind: MSJob  # ms-operator自定的CRD类型，为MSJob
metadata:
  name: ms-widedeep-dataparallel  # 任务名
spec:
  runPolicy: # RunPolicy 封装了分布式训练作业的各种运行时策略，例如如何清理资源以及作业可以保持活动多长时间。
    cleanPodPolicy: None   # All/Running/None
  successPolicy: AllWorkers # 将MSJob标记为success的条件，默认为空，代表使用默认规则（单worker执行完毕即表示成功）
  msReplicaSpecs:
    Scheduler:
      replicas: 1  # Scheduler数量
      restartPolicy: Never  # 重启策略 Always，OnFailure，Never
      template:
        spec:
          volumes: # 文件挂载，如数据集、网络脚本等
            - name: script-data
              hostPath:
                path: /absolute_path
          containers:
            - name: mindspore # 各个角色中必须有且只有一个mindspore名字的container，可配置containerPort来调整默认端口号（2222），需设置端口name为 msjob-port
              image: mindspore-image-name:tag # mindspore镜像
              imagePullPolicy: IfNotPresent
              command: # 容器启动后的执行命令
                - /bin/bash
                - -c
                - python -s /absolute_path/train_and_eval_distribute.py --device_target="GPU" --epochs=1 --data_path=/absolute_path/criteo_mindrecord  --batch_size=16000
              volumeMounts:
                - mountPath: /absolute_path
                  name: script-data
              env:  # 可配置环境变量
                - name: GLOG_v
                  value: "1"
    Worker:
      replicas: 4 # Worker数量
      restartPolicy: Never
      template:
        spec:
          volumes:
            - name: script-data
              hostPath:
                path: /absolute_path
          containers:
            - name: mindspore
              image: mindspore-image-name:tag # mindspore镜像
              imagePullPolicy: IfNotPresent
              command:
                - /bin/bash
                - -c
                - python -s /absolute_path/train_and_eval_distribute.py --device_target="GPU" --epochs=1 --data_path=/absolute_path/criteo_mindrecord --batch_size=16000
              volumeMounts:
                - mountPath: /absolute_path
                  name: script-data
              env:
                - name: GLOG_v
                  value: "1"
              resources: # 资源限制配置
                limits:
                  nvidia.com/gpu: 1
```

### 常见问题

- 镜像构建过程中若发现gcr.io/distroless/static无法拉取，可参考[issue](https://github.com/anjia0532/gcr.io_mirror/issues/169)。
- 安装部署过程中发现gcr.io/kubebuilder/kube-rbac-proxy无法拉取，参考[issue](https://github.com/anjia0532/gcr.io_mirror/issues/153)。
- 当在GPU中通过k8s调起任务，且需要使用NVIDIA显卡时，需要安装k8s device plugin、nvidia-docker2等环境。
- YAML文件配置项中不要使用下划线。
- 当k8s出现阻塞但是通过pod日志无法明确原因时，通过`kubectl logs $(kubectl get statefulset,pods -o wide --all -namespaces|grep ms-operator-system|awk-F""'{print$2}') -n ms-operator-system`查看pod创建过程的日志。
- 通过pod执行任务，默认会在启动的容器根目录下执行，生成的相关文件都会存放在根目录下，但是如果映射路径只是根目录下的某个目录，那生成的文件不会映射保存到宿主机，建议在正式执行任务之前切换路径到指定目录下，便于保存任务执行过程中产生的文件。
- 容灾场景下，如果出现bindIP failed，建议清理上次训练生成持久化文件。
- 不建议在YAML中直接重定向日志文件，如果需要重定向，请区分不同pod的重定向日志文件名。
- Device上存在残留进程或者有其他进程的时候，可能会因为无法申请全部资源导致pod处于Pending状态，建议用户设置超时策略，避免始终被阻塞。