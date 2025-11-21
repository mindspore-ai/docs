import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

export class FileUtils {
  /**
   * 读取文件内容
   */
  static async readFile(filePath: string): Promise<string> {
    try {
      const content = await fs.promises.readFile(filePath, 'utf8');
      return content;
    } catch (error) {
      throw new Error(`读取文件失败: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * 写入文件内容
   */
  static async writeFile(filePath: string, content: string): Promise<void> {
    try {
      await fs.promises.writeFile(filePath, content, 'utf8');
    } catch (error) {
      throw new Error(`写入文件失败: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * 获取当前活动编辑器的文件路径
   */
  static getCurrentRstFile(): string | undefined {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
      return undefined;
    }

    const document = activeEditor.document;
    if (document.languageId !== 'restructuredtext' && !document.fileName.endsWith('.rst')) {
      return undefined;
    }

    return document.fileName;
  }

  /**
   * 显示文件选择器，只选择 .rst 文件
   */
  static async pickRstFile(): Promise<string | undefined> {
    const options: vscode.OpenDialogOptions = {
      canSelectMany: false,
      openLabel: '选择 RST 文件',
      filters: {
        'reStructuredText 文件': ['rst'],
        '所有文件': ['*']
      }
    };

    const fileUri = await vscode.window.showOpenDialog(options);
    if (fileUri && fileUri[0]) {
      return fileUri[0].fsPath;
    }

    return undefined;
  }

  /**
   * 显示文件选择器，支持多选 .rst 文件
   */
  static async pickRstFiles(): Promise<string[]> {
    const options: vscode.OpenDialogOptions = {
      canSelectMany: true,
      openLabel: '选择 RST 文件（支持多选）',
      filters: {
        'reStructuredText 文件': ['rst'],
        '所有文件': ['*']
      }
    };

    const fileUris = await vscode.window.showOpenDialog(options);
    if (fileUris && fileUris.length > 0) {
      return fileUris.map(uri => uri.fsPath);
    }

    return [];
  }

  /**
   * 检查文件是否存在
   */
  static async fileExists(filePath: string): Promise<boolean> {
    try {
      await fs.promises.access(filePath, fs.constants.F_OK);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * 获取文件名（不含路径）
   */
  static getFileName(filePath: string): string {
    return path.basename(filePath);
  }

  /**
   * 获取相对路径（相对于工作区）
   */
  static getRelativePath(filePath: string): string {
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(vscode.Uri.file(filePath));
    if (workspaceFolder) {
      return path.relative(workspaceFolder.uri.fsPath, filePath);
    }
    return path.basename(filePath);
  }
}