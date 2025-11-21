import * as assert from 'assert';
import * as vscode from 'vscode';
import { VirtualDocProvider } from '../../src/diff/virtualDocProvider';

suite('虚拟文档提供者测试', () => {
  let provider: VirtualDocProvider;

  setup(() => {
    provider = new VirtualDocProvider();
  });

  test('应该能够创建 URI', () => {
    const filePath = '/test/file.rst';
    const originalUri = VirtualDocProvider.createUri('original', filePath);
    const optimizedUri = VirtualDocProvider.createUri('optimized', filePath);

    assert.ok(originalUri.scheme === VirtualDocProvider.getScheme());
    assert.ok(optimizedUri.scheme === VirtualDocProvider.getScheme());
    assert.ok(originalUri.path.includes('original'));
    assert.ok(optimizedUri.path.includes('optimized'));
    assert.ok(originalUri.path.includes(encodeURIComponent(filePath)));
    assert.ok(optimizedUri.path.includes(encodeURIComponent(filePath)));
  });

  test('应该能够设置和获取内容', () => {
    const filePath = '/test/file.rst';
    const uri = VirtualDocProvider.createUri('original', filePath);
    const content = '这是测试内容';

    provider.set(uri, content);
    const retrievedContent = provider.provideTextDocumentContent(uri);

    assert.strictEqual(retrievedContent, content);
  });

  test('应该能够清除内容', () => {
    const filePath = '/test/file.rst';
    const uri = VirtualDocProvider.createUri('original', filePath);
    const content = '这是测试内容';

    provider.set(uri, content);
    assert.strictEqual(provider.provideTextDocumentContent(uri), content);

    provider.clear(uri);
    assert.strictEqual(provider.provideTextDocumentContent(uri), undefined);
  });

  test('应该能够清除所有内容', () => {
    const filePath1 = '/test/file1.rst';
    const filePath2 = '/test/file2.rst';
    const uri1 = VirtualDocProvider.createUri('original', filePath1);
    const uri2 = VirtualDocProvider.createUri('optimized', filePath2);

    provider.set(uri1, '内容1');
    provider.set(uri2, '内容2');

    assert.strictEqual(provider.provideTextDocumentContent(uri1), '内容1');
    assert.strictEqual(provider.provideTextDocumentContent(uri2), '内容2');

    provider.clearAll();

    assert.strictEqual(provider.provideTextDocumentContent(uri1), undefined);
    assert.strictEqual(provider.provideTextDocumentContent(uri2), undefined);
  });

  test('不存在的 URI 应该返回 undefined', () => {
    const filePath = '/test/nonexistent.rst';
    const uri = VirtualDocProvider.createUri('original', filePath);

    const content = provider.provideTextDocumentContent(uri);
    assert.strictEqual(content, undefined);
  });
});