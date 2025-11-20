import { FileUtils } from '../utils/file';

export interface BatchResultItem {
  id: string;
  timestamp: number;
  filePath: string;
  originalText: string;
  optimizedText: string;
}

export function getBatchResultsHtml(results: BatchResultItem[]): string {
  const fileItems = results.map((result, index) => {
    const fileName = FileUtils.getFileName(result.filePath);
    const relativePath = FileUtils.getRelativePath(result.filePath);
    const originalLength = result.originalText.length;
    const optimizedLength = result.optimizedText.length;
    const changePercent = Math.round(((optimizedLength - originalLength) / Math.max(1, originalLength)) * 100);
    const date = new Date(result.timestamp || Date.now());
    const timeStr = `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;

    return `
      <div class="file-item">
        <div class="file-header">
          <input type="checkbox" id="file-${index}" class="file-checkbox" data-filepath="${result.filePath}" data-id="${result.id}" checked>
          <label for="file-${index}" class="file-name" onclick="viewDiff('${result.id}')" title="点击查看差异">${fileName}</label>
        </div>
        <div class="file-sub">
          <span class="file-path" title="相对路径">${relativePath}</span>
          <span class="file-time" title="创建时间">${timeStr}</span>
        </div>
        <div class="file-stats">
          <span class="stat">原文: ${originalLength} 字符</span>
          <span class="stat">优化后: ${optimizedLength} 字符</span>
          <span class="stat ${changePercent >= 0 ? 'positive' : 'negative'}">
            变化: ${changePercent >= 0 ? '+' : ''}${changePercent}%
          </span>
        </div>
      </div>
    `;
  }).join('');

  return `
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RST 批量优化结果</title>
        <style>
            body {
                font-family: var(--vscode-font-family);
                font-size: var(--vscode-font-size);
                color: var(--vscode-foreground);
                background-color: var(--vscode-editor-background);
                margin: 0;
                padding: 12px;
                line-height: 1.5;
            }
            .header { margin-bottom: 12px; }
            .title { font-size: 16px; font-weight: bold; margin-bottom: 6px; }
            .summary { color: var(--vscode-descriptionForeground); font-size: 12px; }
            .file-list { margin-bottom: 12px; }
            .file-item { background-color: var(--vscode-list-hoverBackground); border: 1px solid var(--vscode-panel-border); border-radius: 6px; margin-bottom: 8px; padding: 10px; }
            .file-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; min-width: 0; }
            .file-name { font-weight: bold; color: var(--vscode-textLink-foreground); cursor: pointer; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
            .file-sub { display: flex; gap: 8px; color: var(--vscode-descriptionForeground); font-size: 11px; margin: 0 0 6px 22px; overflow: hidden; }
            .file-path { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
            .file-time { white-space: nowrap; }
            .file-stats { display: flex; gap: 8px; font-size: 11px; color: var(--vscode-descriptionForeground); margin-left: 22px; }
            .stat.positive { color: var(--vscode-gitDecoration-addedResourceForeground); }
            .stat.negative { color: var(--vscode-gitDecoration-deletedResourceForeground); }
            .actions { display: flex; gap: 8px; padding-top: 8px; border-top: 1px solid var(--vscode-panel-border); flex-wrap: wrap; }
            .action-btn { padding: 6px 10px; border: none; border-radius: 6px; font-size: 12px; cursor: pointer; transition: all 0.2s ease; }
            .apply-all-btn { background-color: var(--vscode-button-background); color: var(--vscode-button-foreground); }
            .apply-all-btn:hover { background-color: var(--vscode-button-hoverBackground); }
            .apply-selected-btn { background-color: var(--vscode-button-secondaryBackground); color: var(--vscode-button-secondaryForeground); border: 1px solid var(--vscode-button-border); }
            .apply-selected-btn:hover { background-color: var(--vscode-button-secondaryHoverBackground); }
            .discard-btn { background-color: transparent; color: var(--vscode-descriptionForeground); border: 1px solid var(--vscode-button-border); }
            .discard-btn:hover { background-color: var(--vscode-list-hoverBackground); }
            .select-controls { margin: 8px 0; }
            .select-btn { background: none; border: none; color: var(--vscode-textLink-foreground); cursor: pointer; text-decoration: underline; margin-right: 6px; font-size: 11px; }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="title">RST 批量优化结果</div>
            <div class="summary">成功优化了 ${results.length} 个文件，请查看结果并选择要应用的文件</div>
        </div>
        <div class="select-controls">
            <button class="select-btn" onclick="selectAll()">全选</button>
            <button class="select-btn" onclick="selectNone()">全不选</button>
        </div>
        <div class="file-list">${fileItems}</div>
        <div class="actions">
            <button class="action-btn apply-all-btn" onclick="applyAll()">✅ 应用所有更改</button>
            <button class="action-btn apply-selected-btn" onclick="applySelected()">📝 应用选中文件</button>
            <button class="action-btn apply-selected-btn" onclick="deleteSelected()">🗑️ 删除选中记录</button>
            <button class="action-btn discard-btn" onclick="discard()">❌ 关闭</button>
        </div>
        <script>
            const vscode = acquireVsCodeApi();
            function applyAll() { vscode.postMessage({ command: 'applyAll' }); }
            function applySelected() {
                const selectedFiles = Array.from(document.querySelectorAll('.file-checkbox:checked')).map(cb => cb.dataset.filepath);
                if (!selectedFiles.length) { alert('请至少选择一个文件'); return; }
                vscode.postMessage({ command: 'applySelected', selectedFiles });
            }
            function discard() { vscode.postMessage({ command: 'discard' }); }
            function viewDiff(id) { vscode.postMessage({ command: 'viewDiff', id }); }
            function deleteItem(id) { vscode.postMessage({ command: 'deleteItem', id }); }
            function deleteSelected() {
                const selectedIds = Array.from(document.querySelectorAll('.file-checkbox:checked')).map(cb => cb.dataset.id);
                if (!selectedIds.length) { alert('请至少选择一个记录'); return; }
                vscode.postMessage({ command: 'deleteSelected', ids: selectedIds });
            }
            function selectAll() { document.querySelectorAll('.file-checkbox').forEach(cb => cb.checked = true); }
            function selectNone() { document.querySelectorAll('.file-checkbox').forEach(cb => cb.checked = false); }
        </script>
    </body>
    </html>
  `;
}
