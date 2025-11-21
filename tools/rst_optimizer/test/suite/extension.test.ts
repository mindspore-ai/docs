import * as assert from 'assert';
import * as vscode from 'vscode';
import { ConfigManager } from '../../src/config';

suite('扩展测试套件', () => {
  vscode.window.showInformationMessage('开始运行所有测试。');

  test('扩展应该被激活', async () => {
    const extension = vscode.extensions.getExtension('undefined_publisher.rst-optimizer');
    assert.ok(extension);
    
    if (!extension.isActive) {
      await extension.activate();
    }
    
    assert.ok(extension.isActive);
  });

  test('所有命令应该被注册', async () => {
    const commands = await vscode.commands.getCommands(true);
    
    const expectedCommands = [
      'rstOptimizer.optimizeCurrent',
      'rstOptimizer.optimizePickFile',
      'rstOptimizer.applyResult',
      'rstOptimizer.openSettings',
      'rstOptimizer.showQuickPick'
    ];

    for (const command of expectedCommands) {
      assert.ok(commands.includes(command), `命令 ${command} 应该被注册`);
    }
  });
});

suite('配置管理测试', () => {
  test('应该能够读取默认配置', () => {
    const config = ConfigManager.getConfig();
    
    assert.strictEqual(config.provider, 'openai-compatible');
    assert.strictEqual(config.baseUrl, 'https://api.openai.com/v1');
    assert.strictEqual(config.model, 'gpt-4o-mini');
    assert.strictEqual(config.maxTokens, 4096);
    assert.strictEqual(config.temperature, 0.3);
    assert.strictEqual(config.neverUploadIfWorkspaceTrusted, false);
    assert.strictEqual(config.rewrapWidth, 0);
  });

  test('应该验证配置错误', () => {
    const invalidConfig = {
      provider: 'openai-compatible',
      baseUrl: '',
      model: '',
      apiKey: '',
      userPrompt: '测试提示',
      maxTokens: -1,
      temperature: 3,
      neverUploadIfWorkspaceTrusted: false,
      rewrapWidth: 0
    };

    const errors = ConfigManager.validateConfig(invalidConfig);
    
    assert.ok(errors.length > 0, '应该检测到配置错误');
    assert.ok(errors.some(e => e.includes('API 基础 URL')), '应该检测到空的基础 URL');
    assert.ok(errors.some(e => e.includes('模型名称')), '应该检测到空的模型名称');
    assert.ok(errors.some(e => e.includes('API 密钥')), '应该检测到空的 API 密钥');
    assert.ok(errors.some(e => e.includes('最大 token')), '应该检测到无效的最大 token');
    assert.ok(errors.some(e => e.includes('温度值')), '应该检测到无效的温度值');
  });
});