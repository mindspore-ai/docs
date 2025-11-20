import * as path from 'path';
import { glob } from 'glob';

export function run(): Promise<void> {
  const testsRoot = path.resolve(__dirname, '..');

  return new Promise((resolve, reject) => {
    glob('**/**.test.js', { cwd: testsRoot })
      .then((files: string[]) => {
        // 动态导入 Mocha
        const Mocha = require('mocha');
        const mocha = new Mocha({
          ui: 'tdd',
          color: true
        });

        // 添加文件到测试套件
        files.forEach((f: string) => mocha.addFile(path.resolve(testsRoot, f)));

        try {
          // 运行 Mocha 测试
          mocha.run((failures: number) => {
            if (failures > 0) {
              reject(new Error(`${failures} 个测试失败。`));
            } else {
              resolve();
            }
          });
        } catch (err) {
          console.error(err);
          reject(err);
        }
      })
      .catch(reject);
  });
}
