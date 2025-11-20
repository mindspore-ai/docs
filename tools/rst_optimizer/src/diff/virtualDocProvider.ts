import * as vscode from 'vscode';

export class VirtualDocProvider implements vscode.TextDocumentContentProvider {
  private static readonly scheme = 'rstopt';
  private contentMap = new Map<string, string>();
  private _onDidChange = new vscode.EventEmitter<vscode.Uri>();

  readonly onDidChange = this._onDidChange.event;

  static createUri(type: 'original' | 'optimized', filePath: string): vscode.Uri {
    const timestamp = Date.now();
    return vscode.Uri.parse(`${VirtualDocProvider.scheme}:${type}/${encodeURIComponent(filePath)}?t=${timestamp}`);
  }

  provideTextDocumentContent(uri: vscode.Uri): string | undefined {
    const key = this.getKeyFromUri(uri);
    return this.contentMap.get(key);
  }

  set(uri: vscode.Uri, content: string): void {
    const key = this.getKeyFromUri(uri);
    this.contentMap.set(key, content);
    this._onDidChange.fire(uri);
  }

  clear(uri: vscode.Uri): void {
    const key = this.getKeyFromUri(uri);
    this.contentMap.delete(key);
  }

  clearAll(): void {
    this.contentMap.clear();
  }

  private getKeyFromUri(uri: vscode.Uri): string {
    return `${uri.path}${uri.query}`;
  }

  static getScheme(): string {
    return VirtualDocProvider.scheme;
  }
}