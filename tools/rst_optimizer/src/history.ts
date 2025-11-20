import * as vscode from 'vscode';
import type { BatchResultItem } from './views/batchResultsHtml';


export class HistoryStore {
  private static context: vscode.ExtensionContext | undefined;
  private static readonly KEY = 'rstOptimizer.history';

  static init(context: vscode.ExtensionContext) {
    this.context = context;
  }

  private static ensureReady() {
    if (!this.context) {
      throw new Error('HistoryStore not initialized');
    }
  }

  static getAll(): BatchResultItem[] {
    this.ensureReady();
    const items = this.context!.globalState.get<BatchResultItem[]>(this.KEY, []);
    return Array.isArray(items) ? items.slice().sort((a, b) => (b.timestamp ?? 0) - (a.timestamp ?? 0)) : [];
  }

  static addMany(items: Omit<BatchResultItem, 'id' | 'timestamp'>[]): BatchResultItem[] {
    this.ensureReady();
    const existing = this.getAll();
    const now = Date.now();
    const withIds: BatchResultItem[] = items.map((it, idx) => ({
      ...it,
      id: `${now}-${idx}-${Math.random().toString(36).slice(2, 8)}`,
      timestamp: now + idx,
    }));
    const next = [...withIds, ...existing];
    this.context!.globalState.update(this.KEY, next);
    return withIds;
  }

  static addOne(item: Omit<BatchResultItem, 'id' | 'timestamp'>): BatchResultItem {
    return this.addMany([item])[0];
  }

  static removeById(id: string): void {
    this.ensureReady();
    const next = this.getAll().filter(it => it.id !== id);
    this.context!.globalState.update(this.KEY, next);
  }

  static getById(id: string): BatchResultItem | undefined {
    return this.getAll().find(it => it.id === id);
  }

  static clearAll(): void {
    this.ensureReady();
    this.context!.globalState.update(this.KEY, []);
  }
}

