# 使用Runtime执行推理(JAVA)

## Android项目引用AAR包

首先将`mindspore-lite-{version}.aar`文件移动到目标module的**libs**目录，然后在目标module的`build.gradle`的`repositories`中添加本地引用目录，最后在`dependencies`中添加aar的依赖，具体如下所示。

```groovy
repositories {
    flatDir {
        dirs 'libs'
    }
}

dependencies {
    implementation(name:'mindspore-lite-{version}-{timestamp}-{versionCode}', ext:'aar')
}
```

> 注意mindspore-lite-{version}-{timestamp}-{versionCode}是aar的文件名，需要将{version}、{timestamp}、{versionCode}替换成对应信息。

## Android项目使用Mindspore Lite推理框架示例

采用Mindspore Lite Java API推理主要包括`读取模型`、`创建配置上下文`、`创建会话`、`图编译`、`输入数据`、`图执行`、`获得输出`、`内存释放`等步骤。

```java
private boolean init(Context context) {
    // Load the .ms model.
    model = new Model();
    if (!model.loadModel(context, "model.ms")) {
        Log.e("MS_LITE", "Load Model failed");
        return false;
    }
    
    // Create and init config.
    MSConfig msConfig = new MSConfig();
    if (!msConfig.init(DeviceType.DT_CPU, 2, CpuBindMode.MID_CPU)) {
        Log.e("MS_LITE", "Init context failed");
        return false;
    }
    
    // Create the mindspore lite session.
    session = new LiteSession();
    if (!session.init(msConfig)) {
        Log.e("MS_LITE", "Create session failed");
        msConfig.free();
        return false;
    }
    msConfig.free();
    
    // Complile graph.
    if (!session.compileGraph(model)) {
        Log.e("MS_LITE", "Compile graph failed");
        model.freeBuffer();
        return false;
    }
    
    // Note: when use model.freeBuffer(), the model can not be complile graph again.
    model.freeBuffer();

    return true;
}

private void DoInference(Context context) {
    // Set input tensor values.
    List<MSTensor> inputs = session.getInputs();
    if (inputs.size() != 1) {
        Log.e("MS_LITE", "Graph should have one input, but got " + inputs.size() + " inputs");
        return;
    }
    byte[] inData = readFileFromAssets(context, "model_inputs.bin");
    inTensor.setData(inData);
    
    // Run graph to infer results.
    if (!session.runGraph()) {
        Log.e("MS_LITE", "Run graph failed");
        return;
    }

    // Get output tensor values.
    List<String> tensorNames = session.getOutputTensorNames();
    Map<String, MSTensor> outputs = session.getOutputMapByTensor();
    Set<Map.Entry<String, MSTensor>> entrys = outputs.entrySet();
    for (String tensorName : tensorNames) {
        MSTensor output = outputs.get(tensorName);
        if (output == null) {
            Log.e("MS_LITE", "Can not find output " + tensorName);
            return;
        }
        float[] results = output.getFloatData();
        
        // Apply infer results.
        ……
    }
}

// Note: we must release the memory at the end, otherwise it will cause the memory leak.
private void free() {
    session.free();
    model.free();
}
```

