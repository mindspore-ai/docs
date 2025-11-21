import * as vscode from 'vscode';

export interface OptimizationConfig {
  provider: string;
  baseUrl: string;
  model: string;
  apiKey: string;
  userPrompt: string;
  maxTokens: number;
  temperature: number;
  neverUploadIfWorkspaceTrusted: boolean;
  rewrapWidth: number;
}

export interface OptimizationRequest {
  text: string;
  userPrompt: string;
  config: OptimizationConfig;
  variables?: Record<string, string>; // e.g. { fileName, filePath, relativePath, fileBaseName, fileExt }
}

export interface OptimizationResult {
  optimizedText: string;
  originalText: string;
  filePath: string;
}

export interface LLMResponse {
  choices: Array<{
    message: {
      content: string;
    };
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface DiffContext {
  originalUri: vscode.Uri;
  optimizedUri: vscode.Uri;
  filePath: string;
  optimizedContent: string;
}
