import * as vscode from 'vscode';
import { getBatchResultsHtml, BatchResultItem } from './batchResultsHtml';
import { HistoryStore } from '../history';

export type BatchResultsHandlers = {
  onApplyAll: (results: BatchResultItem[]) => Promise<void>;
  onApplySelected: (results: BatchResultItem[], selectedFiles: string[]) => Promise<void>;
  onViewDiffById: (id: string) => Promise<void>;
  onDiscard: () => Promise<void>;
};

export class BatchResultsViewProvider implements vscode.WebviewViewProvider {
  public static readonly ViewId = 'rstOptimizer.batchResults';

  private _view?: vscode.WebviewView;
  private _results: BatchResultItem[] = [];
  private _handlers: BatchResultsHandlers;

  constructor(handlers: BatchResultsHandlers) {
    this._handlers = handlers;
  }

  resolveWebviewView(webviewView: vscode.WebviewView): void | Thenable<void> {
    this._view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    this._results = HistoryStore.getAll();
    this.updateHtml();

    webviewView.webview.onDidReceiveMessage(async message => {
      switch (message.command) {
        case 'applyAll':
          await this._handlers.onApplyAll(this._results);
          break;
        case 'applySelected':
          await this._handlers.onApplySelected(this._results, message.selectedFiles || []);
          break;
        case 'viewDiff':
          if (message.id) {
            await this._handlers.onViewDiffById(message.id);
          }
          break;
        case 'deleteItem':
          if (message.id) {
            HistoryStore.removeById(message.id);
            this._results = HistoryStore.getAll();
            this.updateHtml();
          }
          break;
        case 'deleteSelected':
          if (Array.isArray(message.ids) && message.ids.length) {
            for (const id of message.ids) {
              HistoryStore.removeById(id);
            }
            this._results = HistoryStore.getAll();
            this.updateHtml();
          }
          break;
        case 'discard':
          await this._handlers.onDiscard();
          break;
      }
    });
  }

  async showResults(results: Array<Pick<BatchResultItem, 'filePath' | 'originalText' | 'optimizedText'>>) {
    // Append into history store and refresh from it
    if (results && results.length) {
      // results may not carry id/timestamp when passed in; ensure persistence creates them
      const plain = results.map(r => ({ filePath: r.filePath, originalText: r.originalText, optimizedText: r.optimizedText }));
      HistoryStore.addMany(plain);
    }
    this._results = HistoryStore.getAll();
    this.updateHtml();
    await vscode.commands.executeCommand('workbench.view.explorer');
    await vscode.commands.executeCommand('workbench.view.extension.rstOptimizer'); // in case moved later
    await vscode.commands.executeCommand('rstOptimizer.revealBatchResultsView');
  }

  private updateHtml() {
    if (!this._view) return;
    this._view.webview.html = getBatchResultsHtml(this._results);
  }

  reveal() {
    this._view?.show?.(true);
  }

  refreshFromStore() {
    this._results = HistoryStore.getAll();
    this.updateHtml();
  }
}
