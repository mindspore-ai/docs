# 文档优化插件编译与安装指南

## 🔧 环境要求

* [Node.js](https://nodejs.org/) ≥ 16
* [VS Code](https://code.visualstudio.com/) ≥ 1.80
* 已全局安装 [vsce](https://code.visualstudio.com/api/working-with-extensions/publishing-extension)（用于打包插件）

```bash
npm install -g vsce
```

---

## 🛠️ 编译与打包

在项目根目录执行：

```bash
npm install        # 安装依赖
vsce package       # 生成 .vsix 插件包
```

生成的文件形如：

```
rst-optimizer-0.0.1.vsix
```

---

## 💡 本地安装插件

打开 VS Code，运行命令：

```
Extensions: Install from VSIX...
```

或直接在终端执行：

```bash
code --install-extension rst-optimizer-0.0.1.vsix
```

## 插件配置
安装好插件后，直接在 VsCode 中打开设置，找到本插件设置，填入模型名称和 API Key。