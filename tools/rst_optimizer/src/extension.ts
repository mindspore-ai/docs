import * as vscode from 'vscode';
import { ConfigManager } from './config';
import { LLMClient } from './llm/client';
import { VirtualDocProvider } from './diff/virtualDocProvider';
import { DiffActionCodeLensProvider } from './diff/diffActionCodeLensProvider';
import { BatchResultsViewProvider } from './views/batchResultsView';
import type { BatchResultItem } from './views/batchResultsHtml';
import { FileUtils } from './utils/file';
import { TextWrapper } from './utils/wrap';
import { DiffContext } from './types';
import { HistoryStore } from './history';

let outputChannel: vscode.OutputChannel;
let llmClient: LLMClient;
let virtualDocProvider: VirtualDocProvider;
let statusBarItem: vscode.StatusBarItem;
let currentDiffContext: DiffContext | undefined;
let batchResultsViewProvider: BatchResultsViewProvider;

export function activate(context: vscode.ExtensionContext) {
  outputChannel = vscode.window.createOutputChannel('RST Optimizer');
  context.subscriptions.push(outputChannel);

  llmClient = new LLMClient(outputChannel);

  HistoryStore.init(context);

  virtualDocProvider = new VirtualDocProvider();
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider(
      VirtualDocProvider.getScheme(),
      virtualDocProvider
    )
  );

  const codeLensProvider = new DiffActionCodeLensProvider();
  context.subscriptions.push(
    vscode.languages.registerCodeLensProvider(
      { scheme: VirtualDocProvider.getScheme() },
      codeLensProvider
    )
  );

  batchResultsViewProvider = new BatchResultsViewProvider({
    onApplyAll: async (results) => applyAllBatchResults(results),
    onApplySelected: async (results, selected) => applySelectedBatchResults(results, selected),
    onViewDiffById: async (id) => {
      const r = HistoryStore.getById(id);
      if (r) {
        await showDiffView(r.filePath, r.originalText, r.optimizedText);
      }
    },
    onDiscard: async () => {/* no-op, close action handled by view */}
  });
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(BatchResultsViewProvider.ViewId, batchResultsViewProvider)
  );
  // 命令：确保可以显式聚焦视图
  context.subscriptions.push(
    vscode.commands.registerCommand('rstOptimizer.revealBatchResultsView', () => batchResultsViewProvider.reveal())
  );

  // 创建状态栏项
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBarItem.text = '$(edit) RST Optimizer';
  statusBarItem.tooltip = '点击打开 RST Optimizer 命令';
  statusBarItem.command = 'rstOptimizer.showQuickPick';
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  // 注册命令
  const commands = [
    vscode.commands.registerCommand('rstOptimizer.optimizeCurrent', optimizeCurrentFile),
    vscode.commands.registerCommand('rstOptimizer.optimizePickFile', optimizePickedFile),
    vscode.commands.registerCommand('rstOptimizer.applyResult', applyOptimizedResult),
    vscode.commands.registerCommand('rstOptimizer.discardResult', discardOptimizedResult),
    vscode.commands.registerCommand('rstOptimizer.openSettings', openSettings),
    vscode.commands.registerCommand('rstOptimizer.showQuickPick', showQuickPick)
  ];

  context.subscriptions.push(...commands);

  outputChannel.appendLine('RST Optimizer 扩展已激活');
}

export function deactivate() {
  if (virtualDocProvider) {
    virtualDocProvider.clearAll();
  }
  outputChannel?.appendLine('RST Optimizer 扩展已停用');
}

async function optimizeCurrentFile() {
  const filePath = FileUtils.getCurrentRstFile();
  if (!filePath) {
    vscode.window.showWarningMessage('当前编辑器不是 RST 文件，请打开一个 .rst 文件');
    return;
  }

  await optimizeFile(filePath);
}

async function optimizePickedFile() {
  const filePaths = await FileUtils.pickRstFiles();
  if (filePaths.length === 0) {
    return; // 用户取消了选择
  }

  // 打开选中的文件
  for (const filePath of filePaths) {
    const document = await vscode.workspace.openTextDocument(filePath);
    await vscode.window.showTextDocument(document, { preview: false });
  }

  if (filePaths.length === 1) {
    // 单文件优化
    await optimizeFile(filePaths[0]);
  } else {
    // 批量优化
    await optimizeMultipleFiles(filePaths);
  }
}

async function optimizeMultipleFiles(filePaths: string[]) {
  try {
    // 检查工作区安全设置
    if (!ConfigManager.checkWorkspaceSafety()) {
      return;
    }

    // 获取配置
    const config = ConfigManager.getConfig();
    const configErrors = ConfigManager.validateConfig(config);
    if (configErrors.length > 0) {
      vscode.window.showErrorMessage(`配置错误：${configErrors.join(', ')}`);
      return;
    }

    const results: { filePath: string; originalText: string; optimizedText: string }[] = [];
    
    // 显示总体进度
    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `正在批量优化 ${filePaths.length} 个 RST 文件...`,
        cancellable: true
      },
      async (progress, token) => {
        for (let i = 0; i < filePaths.length; i++) {
          const filePath = filePaths[i];
          const fileName = FileUtils.getFileName(filePath);
          
          if (token.isCancellationRequested) {
            break;
          }

          progress.report({
            message: `正在优化 ${fileName} (${i + 1}/${filePaths.length})`,
            increment: (100 / filePaths.length)
          });

          try {
            const originalText = await FileUtils.readFile(filePath);
            if (!originalText.trim()) {
              outputChannel.appendLine(`跳过空文件: ${filePath}`);
              continue;
            }

            const dotIdx = fileName.lastIndexOf('.');
            const fileBaseName = dotIdx > 0 ? fileName.slice(0, dotIdx) : fileName;
            const fileExt = dotIdx > -1 ? fileName.slice(dotIdx + 1) : '';
            const optimizedText = await llmClient.optimizeRst(
              {
                text: originalText,
                userPrompt: config.userPrompt,
                config,
                variables: {
                  filePath,
                  fileName,
                  relativePath: FileUtils.getRelativePath(filePath),
                  fileBaseName,
                  fileExt
                }
              },
              token
            );

            const finalOptimizedText = config.rewrapWidth > 0 
              ? TextWrapper.wrapText(optimizedText, config.rewrapWidth)
              : optimizedText;

            results.push({
              filePath,
              originalText,
              optimizedText: finalOptimizedText
            });

          } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            outputChannel.appendLine(`优化失败 ${filePath}: ${errorMessage}`);
            vscode.window.showWarningMessage(`优化 ${fileName} 失败: ${errorMessage}`);
          }
        }
      }
    );

    if (results.length > 0) {
      await showBatchOptimizationResults(results);
    } else {
      vscode.window.showWarningMessage('没有成功优化任何文件');
    }

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    vscode.window.showErrorMessage(`批量优化失败: ${errorMessage}`);
    outputChannel.appendLine(`批量优化失败: ${errorMessage}`);
  }
}

async function optimizeFile(filePath: string) {
  try {
    // 检查工作区安全设置
    if (!ConfigManager.checkWorkspaceSafety()) {
      return;
    }

    // 获取配置
    const config = ConfigManager.getConfig();
    const configErrors = ConfigManager.validateConfig(config);
    if (configErrors.length > 0) {
      vscode.window.showErrorMessage(`配置错误：${configErrors.join(', ')}`);
      return;
    }

    // 读取文件内容
    const originalText = await FileUtils.readFile(filePath);
    if (!originalText.trim()) {
      vscode.window.showWarningMessage('文件内容为空');
      return;
    }

    // 显示进度并执行优化
    const optimizedText = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `正在使用 ${config.provider}/${config.model} 优化 RST 文档...`,
        cancellable: true
      },
      async (progress, token) => {
        const fileName = FileUtils.getFileName(filePath);
        const relativePath = FileUtils.getRelativePath(filePath);
        const dotIdx = fileName.lastIndexOf('.');
        const fileBaseName = dotIdx > 0 ? fileName.slice(0, dotIdx) : fileName;
        const fileExt = dotIdx > -1 ? fileName.slice(dotIdx + 1) : '';
        return await llmClient.optimizeRst(
          {
            text: originalText,
            userPrompt: config.userPrompt,
            config,
            variables: { filePath, fileName, relativePath, fileBaseName, fileExt }
          },
          token
        );
      }
    );

    // 应用文本包装（如果配置了）
    const finalOptimizedText = config.rewrapWidth > 0 
      ? TextWrapper.wrapText(optimizedText, config.rewrapWidth)
      : optimizedText;

    // 创建 diff 视图
    await showDiffView(filePath, originalText, finalOptimizedText);

    // 记录到历史并刷新侧边栏
    await batchResultsViewProvider.showResults([{ filePath, originalText, optimizedText: finalOptimizedText }]);

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    vscode.window.showErrorMessage(`优化失败: ${errorMessage}`);
    outputChannel.appendLine(`优化失败: ${errorMessage}`);
  }
}

async function showDiffView(filePath: string, originalText: string, optimizedText: string) {
  // 创建虚拟 URI
  const originalUri = VirtualDocProvider.createUri('original', filePath);
  const optimizedUri = VirtualDocProvider.createUri('optimized', filePath);

  // 设置虚拟文档内容
  virtualDocProvider.set(originalUri, originalText);
  virtualDocProvider.set(optimizedUri, optimizedText);

  // 保存当前 diff 上下文
  currentDiffContext = {
    originalUri,
    optimizedUri,
    filePath,
    optimizedContent: optimizedText
  };

  // 打开 diff 视图
  const fileName = FileUtils.getFileName(filePath);
  const title = `RST Diff: ${fileName}`;
  
  await vscode.commands.executeCommand(
    'vscode.diff',
    originalUri,
    optimizedUri,
    title,
    { preview: false }
  );

  // 设置上下文，确保菜单按钮在 diff 标题栏显示
  await vscode.commands.executeCommand('setContext', 'rstOptimizer.hasActiveDiff', true);
}

async function showBatchOptimizationResults(results: { filePath: string; originalText: string; optimizedText: string }[]) {
  await batchResultsViewProvider.showResults(results);
}


async function applyOptimizedResult() {
  if (!currentDiffContext) {
    vscode.window.showWarningMessage('没有可应用的优化结果');
    return;
  }

  try {
    await FileUtils.writeFile(currentDiffContext.filePath, currentDiffContext.optimizedContent);
    
    const fileName = FileUtils.getFileName(currentDiffContext.filePath);
    vscode.window.showInformationMessage(`已成功应用优化结果到 ${fileName}`);
    
    outputChannel.appendLine(`[${new Date().toISOString()}] 已应用优化结果: ${currentDiffContext.filePath}`);
    
    // 清理
    await discardChanges();
    
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    vscode.window.showErrorMessage(`应用更改失败: ${errorMessage}`);
  }
}

async function discardOptimizedResult() {
  if (!currentDiffContext) {
    vscode.window.showWarningMessage('没有可放弃的优化结果');
    return;
  }

  const fileName = FileUtils.getFileName(currentDiffContext.filePath);
  vscode.window.showInformationMessage(`已放弃对 ${fileName} 的优化结果`);
  
  outputChannel.appendLine(`[${new Date().toISOString()}] 已放弃优化结果: ${currentDiffContext.filePath}`);
  
  // 清理
  await discardChanges();
}

async function discardChanges() {
  if (currentDiffContext) {
    // 清理虚拟文档
    virtualDocProvider.clear(currentDiffContext.originalUri);
    virtualDocProvider.clear(currentDiffContext.optimizedUri);
    currentDiffContext = undefined;
  }
  // 清理上下文
  await vscode.commands.executeCommand('setContext', 'rstOptimizer.hasActiveDiff', false);
}

async function applyAllBatchResults(results: { filePath: string; originalText: string; optimizedText: string }[]) {
  let successCount = 0;
  let failCount = 0;

  for (const result of results) {
    try {
      await FileUtils.writeFile(result.filePath, result.optimizedText);
      successCount++;
      outputChannel.appendLine(`[${new Date().toISOString()}] 已应用优化结果: ${result.filePath}`);
    } catch (error) {
      failCount++;
      const errorMessage = error instanceof Error ? error.message : String(error);
      outputChannel.appendLine(`[${new Date().toISOString()}] 应用失败: ${result.filePath} - ${errorMessage}`);
    }
  }

  vscode.window.showInformationMessage(
    `批量应用完成！成功: ${successCount} 个，失败: ${failCount} 个`
  );
}

async function applySelectedBatchResults(
  results: { filePath: string; originalText: string; optimizedText: string }[],
  selectedFiles: string[]
) {
  let successCount = 0;
  let failCount = 0;

  for (const filePath of selectedFiles) {
    const result = results.find(r => r.filePath === filePath);
    if (!result) continue;

    try {
      await FileUtils.writeFile(result.filePath, result.optimizedText);
      successCount++;
      outputChannel.appendLine(`[${new Date().toISOString()}] 已应用优化结果: ${result.filePath}`);
    } catch (error) {
      failCount++;
      const errorMessage = error instanceof Error ? error.message : String(error);
      outputChannel.appendLine(`[${new Date().toISOString()}] 应用失败: ${result.filePath} - ${errorMessage}`);
    }
  }

  vscode.window.showInformationMessage(
    `选择性应用完成！成功: ${successCount} 个，失败: ${failCount} 个`
  );
}

// batch 结果的 HTML 已移动到 src/views/batchResultsHtml.ts

async function openSettings() {
  await vscode.commands.executeCommand('workbench.action.openSettings', 'rstOptimizer');
}

async function showQuickPick() {
  const items = [
    {
      label: '$(file-text) 优化当前 RST 文件',
      description: '优化当前活动编辑器中的 RST 文件',
      command: 'rstOptimizer.optimizeCurrent'
    },
    {
      label: '$(folder-opened) 选择 RST 文件优化',
      description: '通过文件选择器选择要优化的 RST 文件',
      command: 'rstOptimizer.optimizePickFile'
    },
    {
      label: '$(settings-gear) 打开设置',
      description: '配置 RST Optimizer 设置',
      command: 'rstOptimizer.openSettings'
    }
  ];

  const selected = await vscode.window.showQuickPick(items, {
    placeHolder: '选择要执行的操作'
  });

  if (selected) {
    await vscode.commands.executeCommand(selected.command);
  }
}
