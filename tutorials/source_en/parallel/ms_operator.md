# Performing Distributed Training on K8S Clusters

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_en/parallel/ms_operator.md)

MindSpore Operator is a plugin that follows Kubernetes' Operator pattern (based on the CRD-Custom Resource Definition feature) and implements distributed training on Kubernetes. MindSpore Operator defines Scheduler, PS, worker three roles in CRD, and users can easily use MindSpore on K8S for distributed training through simple YAML file configuration. The code repository of mindSpore Operator is described in: [ms-operator](https://gitee.com/mindspore/ms-operator/).

## Installation

There are three installation methods:

1. Install directly by using YAML

   ```shell
   kubectl apply -f deploy/v1/ms-operator.yaml
   ```

   After installation:

   Use `kubectl get pods --all-namespaces` to see the namespace as the deployment task for the ms-operator-system.

   Use `kubectl describe pod ms-operator-controller-manager-xxx-xxx -n ms-operator-system` to view pod details.

2. Install by using make deploy

   ```shell
   make deploy IMG=swr.cn-south-1.myhuaweicloud.com/mindspore/ms-operator:latest
   ```

3. Local debugging environment

   ```shell
   make run
   ```

## Sample

The current ms-operator supports ordinary single worker training, single worker training in PS mode, and Scheduler and Worker startups for automatic parallelism (such as data parallelism and model parallelism).

There are running examples in [config/samples/](https://gitee.com/mindspore/ms-operator/tree/master/config/samples). Take the data-parallel Scheduler and Worker startup as an example, where the dataset and network scripts need to be prepared in advance:

```shell
kubectl apply -f config/samples/ms_wide_deep_dataparallel.yaml
```

Use `kubectl get all -o wide` to see scheduler and worker launched in the cluster, as well as the services corresponding to Scheduler.

## Development Guide

### Core Code

`pkg/apis/v1/msjob_types.go` is the CRD definition for MSJob.

`pkg/controllers/v1/msjob_controller.go` is the core logic of the MSJob controller.

### Image Creation and Uploading

To modify the ms-operator code and create an upload image, please refer to the following command:

```shell
make docker-build IMG={image_name}:{tag}
docker push {image_name}:{tag}
```

### YAML File Configuration Instructions

Taking the data parallelization of self-developed networking as an example, the YAML configuration of MSJob is introduced, such as `runPolicy`, `successPolicy`, the number of roles, mindspore images, and file mounting, and users need to configure it according to their actual needs.

```yaml
apiVersion: mindspore.gitee.com/v1
kind: MSJob  # ms-operator custom CRD type, MSJob
metadata:
  name: ms-widedeep-dataparallel  # Task name
spec:
  runPolicy: # RunPolicy encapsulates various runtime strategies for distributed training jobs, such as how to clean up resources and how long the job can remain active.
    cleanPodPolicy: None   # All/Running/None
  successPolicy: AllWorkers # The condition that marks MSJob as subcess, which defaults to blank, represents the use of the default rule (success after a single worker execution is completed)
  msReplicaSpecs:
    Scheduler:
      replicas: 1  # The number of Scheduler
      restartPolicy: Never  # Restart the policy Always, OnFailure, Never
      template:
        spec:
          volumes: # File mounts, such as datasets, network scripts, and so on
            - name: script-data
              hostPath:
                path: /absolute_path
          containers:
            - name: mindspore # Each character must have a container with only one mindspore name, configure containerPort to adjust the default port number (2222), and you need to set the port name to msjob-port
              image: mindspore-image-name:tag # mindspore image
              imagePullPolicy: IfNotPresent
              command: # Execute the command after the container starts
                - /bin/bash
                - -c
                - python -s /absolute_path/train_and_eval_distribute.py --device_target="GPU" --epochs=1 --data_path=/absolute_path/criteo_mindrecord  --batch_size=16000
              volumeMounts:
                - mountPath: /absolute_path
                  name: script-data
              env:  # Configurable environment variables
                - name: GLOG_v
                  value: "1"
    Worker:
      replicas: 4 # The number of Worker
      restartPolicy: Never
      template:
        spec:
          volumes:
            - name: script-data
              hostPath:
                path: /absolute_path
          containers:
            - name: mindspore
              image: mindspore-image-name:tag # mindspore image
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
              resources: # Resource limit configuration
                limits:
                  nvidia.com/gpu: 1
```

### Frequent Questions

- If you find that gcr.io/distroless/static cannot be pulled during the image construction process, see [issue](https://github.com/anjia0532/gcr.io_mirror/issues/169).
- During the installation and deployment process, when finding that the gcr.io/kubebuilder/kube-rbac-proxy cannot be pulled, see [issue](https://github.com/anjia0532/gcr.io_mirror/issues/153).
- When you call up tasks through k8s in the GPU and need to use NVIDIA graphics cards, you need to install k8s device plugin, nvidia-docker2 and other environments.
- Do not use underscores in YAML file configuration items.
- When k8s is blocked but the cause cannot be determined by the pod log, view the log of the pod creation process via `kubectl logs $(kubectl get statefulset,pods -o wide --all -namespaces|grep ms-operator-system|awk-F""'{print$2}') -n ms-operator-system`.
- Performing tasks through the pod, it will be executed in the root directory of the launched container, and the relevant files generated will be stored in the root directory by default. But if the mapping path is only a directory under the root directory, the generated files will not be mapped and saved to the host. It is recommended to switch the path to the specified directory before officially performing the task, so as to save the files generated during the execution of the task.
- In the disaster recovery scenario, if bindIP failed occurs, confirm whether the persistence file generated by the last training has not been cleaned.
- It is not recommended to redirect log files directly in YAML. If redirection is required, distinguish between redirect log file names for different pods.
- When there are residual processes or other processes on the Device, the pod may be in Pending state due to the inability to apply for all the resources, and it is recommended that the user set a timeout strategy to avoid being blocked all the time.
