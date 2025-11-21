import * as vscode from 'vscode';
import { OptimizationRequest, LLMResponse } from '../types';

const DEFAULT_SYSTEM_PROMPT = `你是 reStructuredText（RST）与 Sphinx 文档优化专家。目标：在不破坏语义与构建的前提下，让文档更清晰专业。
严格遵循：
1) 保留并尊重所有 Sphinx 角色/指令/域（如 :ref:、:class:、:func:、.. code-block::、.. note::、.. figure:: 等），不要更改其语法结构与缩进。
2) 绝不删除或更改链接目标、交叉引用锚点（如 .. _anchor:）。
3) 保留代码块、控制台示例与行内字面值（\`\`literal\`\`）原样；除非是修复明显拼写错误。
4) 标题层级与下划线风格统一（如 # * = - ^ ~ 等），但不得改变层级关系。
5) 修复语法/拼写/术语一致性，使段落更简洁、技术准确；尽量减少被动语态。
6) 表格、列表、缩进必须合法；指令体的缩进四空格对齐。
7) 输出**完整优化后的整篇 RST**，不要输出 diff 或注释。`;

export class LLMClient {
  private outputChannel: vscode.OutputChannel;

  constructor(outputChannel: vscode.OutputChannel) {
    this.outputChannel = outputChannel;
  }

  async optimizeRst(
    request: OptimizationRequest,
    cancellationToken?: vscode.CancellationToken
  ): Promise<string> {
    const startTime = Date.now();
    const { text, userPrompt, config } = request;

    this.outputChannel.appendLine(`[${new Date().toISOString()}] 开始优化 RST 文档`);
    this.outputChannel.appendLine(`提供商: ${config.provider}`);
    this.outputChannel.appendLine(`模型: ${config.model}`);
    this.outputChannel.appendLine(`文档长度: ${text.length} 字符`);
    this.outputChannel.appendLine(`估算 token: ${Math.ceil(text.length / 4)}`);

    try {
      const userContent = this.composeUserContent(text, userPrompt, request.variables);
      
      const requestBody = {
        model: config.model,
        temperature: config.temperature,
        max_tokens: config.maxTokens,
        messages: [
          { role: "system", content: DEFAULT_SYSTEM_PROMPT },
          { role: "user", content: userContent }
        ]
      };

      const controller = new AbortController();
      
      // 处理取消令牌
      if (cancellationToken) {
        cancellationToken.onCancellationRequested(() => {
          controller.abort();
          this.outputChannel.appendLine(`[${new Date().toISOString()}] 请求被用户取消`);
        });
      }

      const response = await fetch(`${config.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${config.apiKey}`
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const result = await response.json() as LLMResponse;
      
      if (!result.choices || result.choices.length === 0) {
        throw new Error('API 返回了空的选择列表');
      }

      const optimizedText = result.choices[0].message.content;
      const endTime = Date.now();
      const duration = endTime - startTime;

      this.outputChannel.appendLine(`[${new Date().toISOString()}] 优化完成`);
      this.outputChannel.appendLine(`耗时: ${duration}ms`);
      
      if (result.usage) {
        this.outputChannel.appendLine(`Token 使用: ${result.usage.prompt_tokens} + ${result.usage.completion_tokens} = ${result.usage.total_tokens}`);
      }

      return optimizedText;

    } catch (error) {
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      this.outputChannel.appendLine(`[${new Date().toISOString()}] 优化失败 (${duration}ms)`);
      
      if (error instanceof Error) {
        this.outputChannel.appendLine(`错误: ${error.message}`);
        if (error.stack) {
          this.outputChannel.appendLine(`堆栈: ${error.stack}`);
        }
      } else {
        this.outputChannel.appendLine(`未知错误: ${String(error)}`);
      }

      throw error;
    }
  }

  private composeUserContent(text: string, userPrompt: string, variables?: Record<string, string>): string {
    const replaced = variables
      ? userPrompt.replace(/\$\{(\w+)\}/g, (_m, k) => (variables[k] ?? `
${'${'}${k}}`))
      : userPrompt;
    return `${replaced}\n\n以下是需要优化的 RST 文档内容：\n\n${text}`;
  }
}
