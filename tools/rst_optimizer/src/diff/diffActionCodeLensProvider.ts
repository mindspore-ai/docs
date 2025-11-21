import * as vscode from 'vscode';
import { VirtualDocProvider } from './virtualDocProvider';


export class DiffActionCodeLensProvider implements vscode.CodeLensProvider {
  private onDidChangeCodeLensesEmitter = new vscode.EventEmitter<void>();
  public readonly onDidChangeCodeLenses: vscode.Event<void> = this.onDidChangeCodeLensesEmitter.event;

  provideCodeLenses(document: vscode.TextDocument): vscode.ProviderResult<vscode.CodeLens[]> {
    if (document.uri.scheme !== VirtualDocProvider.getScheme()) {
      return [];
    }
    if (!document.uri.path.startsWith('/optimized/')) {
      return [];
    }

    const lastLine = Math.max(0, document.lineCount - 1);
    const range = new vscode.Range(lastLine, 0, lastLine, 0);

    const applyLens = new vscode.CodeLens(range, {
      title: '$(check) 应用优化',
      command: 'rstOptimizer.applyResult'
    });

    const discardLens = new vscode.CodeLens(range, {
      title: '$(x) 放弃优化',
      command: 'rstOptimizer.discardResult'
    });

    return [applyLens, discardLens];
  }

  refresh(): void {
    this.onDidChangeCodeLensesEmitter.fire();
  }
}

