export class TextWrapper {
  /**
   * 对文本进行硬换行包裹
   * @param text 原始文本
   * @param width 行宽限制
   * @returns 包裹后的文本
   */
  static wrapText(text: string, width: number): string {
    if (width <= 0) {
      return text;
    }

    const lines = text.split('\n');
    const wrappedLines: string[] = [];

    for (const line of lines) {
      // 保留空行
      if (line.trim() === '') {
        wrappedLines.push(line);
        continue;
      }

      // 检查是否是 RST 特殊行（不应该被包裹）
      if (this.shouldPreserveLine(line)) {
        wrappedLines.push(line);
        continue;
      }

      // 对普通文本行进行包裹
      const wrapped = this.wrapLine(line, width);
      wrappedLines.push(...wrapped);
    }

    return wrappedLines.join('\n');
  }

  /**
   * 检查行是否应该保持原样（不进行包裹）
   */
  private static shouldPreserveLine(line: string): boolean {
    const trimmed = line.trim();
    
    // RST 指令行
    if (trimmed.startsWith('.. ')) {
      return true;
    }

    // 标题下划线
    if (/^[\s]*[=\-`:'~^_*+#<>"]{3,}[\s]*$/.test(trimmed)) {
      return true;
    }

    // 代码块内容（通过缩进判断）
    if (line.startsWith('    ') || line.startsWith('\t')) {
      return true;
    }

    // 列表项
    if (/^[\s]*[-*+]\s/.test(line) || /^[\s]*\d+\.\s/.test(line)) {
      return true;
    }

    // 表格行
    if (trimmed.includes('|') && trimmed.length > 10) {
      return true;
    }

    // 链接定义
    if (/^[\s]*\.\. _[^:]+:/.test(line)) {
      return true;
    }

    return false;
  }

  /**
   * 包裹单行文本
   */
  private static wrapLine(line: string, width: number): string[] {
    const leadingWhitespace = line.match(/^(\s*)/)?.[1] || '';
    const content = line.trim();

    if (content.length <= width - leadingWhitespace.length) {
      return [line];
    }

    const words = content.split(/\s+/);
    const wrappedLines: string[] = [];
    let currentLine = leadingWhitespace;

    for (const word of words) {
      const testLine = currentLine === leadingWhitespace 
        ? currentLine + word 
        : currentLine + ' ' + word;

      if (testLine.length <= width) {
        currentLine = testLine;
      } else {
        if (currentLine.trim()) {
          wrappedLines.push(currentLine);
        }
        currentLine = leadingWhitespace + word;
      }
    }

    if (currentLine.trim()) {
      wrappedLines.push(currentLine);
    }

    return wrappedLines.length > 0 ? wrappedLines : [line];
  }
}