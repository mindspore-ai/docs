import * as vscode from 'vscode';
import { OptimizationConfig } from './types';

export class ConfigManager {
  private static readonly CONFIG_SECTION = 'rstOptimizer';

  static getConfig(): OptimizationConfig {
    const config = vscode.workspace.getConfiguration(this.CONFIG_SECTION);
    
    // 优先从环境变量读取 API Key
    const apiKey = process.env.RST_OPTIMIZER_API_KEY || config.get<string>('api.apiKey', '');
    
    return {
      provider: config.get<string>('api.provider', 'openai-compatible'),
      baseUrl: config.get<string>('api.baseUrl', 'https://api.openai.com/v1'),
      model: config.get<string>('api.model', 'gpt-4o-mini'),
      apiKey,
      userPrompt: config.get<string>('prompt.userPrompt',"你是一名 **RST 技术文档审校与修复专家**。\n你的任务是：在**不改变技术语义**的前提下，**严格检查并修复**给定 RST 文档在语法、结构、格式与中文排版上的问题，并**保持与 API 定义及文件名的一致性**。\n如无明确要求，**不要引入新的内容段落或编造默认值/异常**；仅在缺漏“必须存在且可从上下文直接确定”的字段时做最小增补。\n\n## 输入\n\n* 文件名：${fileName}\n* 原始 RST 文本（保持原样）\n\n## 总体原则\n\n1. **最小侵入修复**：仅修复错误或不规范之处；保留原有信息结构与术语。\n2. **禁止误改代码/公式**：`code-block`、`literalinclude`、`math` 等指令体及反引号行内代码的技术内容不得改写（仅可做空格与围栏修复）。\n3. **一致性优先**：文件名、章节标题、API 名称必须一致（见规则 2）。\n4. **中文排版**：面向中文读者的正文需优化中文标点与用语，但不得改变技术意义。\n5. **不可臆测**：若无默认值/异常/类型信息，不得凭空添加。无法确定时保持空或原状，并在总结里标注“需要人工确认”。\n\n## 必查必修规则（逐条执行）\n\n### 1）通用 RST 语法与特殊标记\n\n* 检查并修复常见指令语法：`.. note::`、`.. warning::`、`.. seealso::`、`.. include::`、`.. math::` 等的缩进与空行：\n  * 指令与其内容之间需空一行（除确有嵌套要求的场景外）。\n  * 指令体内容相对指令行统一缩进（建议 3–4 空格），全文保持一致。\n* 行内特殊标记规则：\n  * `*斜体*`、`**加粗**`、``行内代码``：标记内部不得出现空格（确需空格时改为转义或拆分）。\n  * 标记与上下文：标记前后各保留 1 个空格（句首或紧邻标点处除外）。\n* 修复多余反引号、未闭合标记、错误嵌套。\n\n### 2）文件名 / 章节标题 / API 定义名一致性\n\n* 要求：文件名、文档首个最高层级标题、正文首个 API 定义名应一致。\n* 例外：若文件名匹配 `mindspore.xxx.func_yyy.rst`，则章节标题与 API 定义名统一为 `mindspore.xxx.yyy`（去掉 `func_`）。\n* 实施：\n  * 若三者不一致，以 API 定义名为准统一章节标题；遇到上述例外时按例外规则处理。\n  * 若文件名与 API 命名空间明显冲突（如包路径不同），不改文件名，仅统一标题与文内 API 显示。\n\n### 3）“参数/关键字参数”模块格式（**包含关键字专用参数的严格规则**）\n\n* 目标格式（逐项以无序列表 `-` 起）：\n\n```\n参数：\n- **参数名** (数据类型[, 可选]) – 参数说明。默认值：`None`，表示……\n  关键字参数：\n- **参数名** (数据类型[, 可选]) – 参数说明。默认值：`None`，表示……\n```\n\n* 严格对齐 Python 函数签名的五类参数并分流到正确模块：\n\n1) **仅位置参数（positional-only）**：位于 `/` 左侧（若签名包含 `/`）。归入“参数”模块，**不要展示 `/` 本身**。\n2) **位置或关键字参数（positional-or-keyword）**：常见的 `name` 或 `name=...`。归入“参数”模块。\n3) **可变位置参数**：`*args`。归入“参数”模块，名称保留星号前缀 `*`，类型与说明按项目规范书写。\n4) **关键字专用参数（keyword-only）**：当签名中出现 **裸 `*` 分隔符** 或 `*args` 之后的形参（直到 `**kwargs` 之前）。**这些形参必须归入“关键字参数”模块**。\n   - 裸 `*` 仅是分隔符，**不出现在文档中**；其右侧的参数（如 `dtype=None`）统一移至“关键字参数”。\n5) **可变关键字参数**：`**kwargs`。根据项目约定决定是否在“关键字参数”模块列出；若列出需保留 `**` 前缀并简要说明用途，避免臆测具体键。\n\n* “参数/关键字参数”的**名称与顺序必须与函数定义完全一致**（先左后右、从签名原序列化得到）。\n* “可选/默认值”标注规则：\n* 形参**有默认值**（含 `=None`）或注解为可选类型时：在类型后补 `, 可选`，且在说明末尾追加“默认值：``<literal>``，……”。\n* 形参**无默认值**：不写“可选”，不写默认值句。\n* **数据类型**需与定义一致；若原文缺失且上下文也无法确定，不要臆测，类型可暂缺省或保持原状。\n* 术语与标点：\n* 中文破折号统一使用 `–`（en dash），全文保持一致。\n* 默认值文字中的字面量使用行内代码围栏（如 ``None``、``True``、``-1``）。\n* **迁移示例（与你给的 case 对齐）**：签名 `(..., dim=None, *, dtype=None)` →\n  `dim` 仍在“参数”；`dtype` 作为 **关键字专用参数** 移至“关键字参数”。\n\n### 4）“异常”模块格式\n\n* 目标格式：\n\n```\n异常：\n- **ErrorType** - 异常描述。\n```\n\n* 同类异常归并在一起，子类在前、父类在后；不得杜撰异常，无法确认时保留原状并在总结中标注需确认。\n\n### 5）“输入/输出”模块格式\n\n* 目标格式：\n\n```\n输入：\n- **输入名** (数据类型) – 描述。\n  输出：\n- **输出名** (数据类型) – 描述。\n```\n\n* 无序列表 `-`，加粗名称 + 圆括号数据类型 + `–` 描述。\n\n### 6）换行与缩进\n\n* 普通段落换行不缩进。\n* 有序/无序列表换行统一 2 个空格缩进，与上一行正文起始位置对齐。\n* 指令体（note/warning/seealso/include/math 等）内层统一缩进；子块（如代码）再按 RST 规范缩进一级。\n\n### 7）模块间空行\n\n* 各模块之间至少 1 个空行；指令与其前后段落之间保留空行，避免黏连。\n\n### 8）中文文本与排版\n\n* 修复错别字、冗余空格、英文标点混用（中文语句中使用中文标点：`，` `。` `：` `；` `（ ）`）。\n* 并列关系用顿号 `、` 分隔。\n* API 名、类名、参数名、代码标识符保留原文并用行内代码``包裹。\n* 句子更通顺简洁，但不得改变技术含义。\n\n## 额外一致性检查\n\n* 标题层级符号（`= - ~ ^ \" ' *` 等）长度需与标题文本长度一致；同级标题符号统一。\n* `:param:/:type:/:return:/:rtype:` 若与目标“列表式参数节”并存，应二选一统一为项目约定风格（通常统一为“参数/关键字参数”块）。\n* 交叉引用（`:class:`, `:func:`, `:mod:` 等）语法修复：角色名、目标、反引号与空格。\n* `include` 路径前后空格与相对路径的一致性。\n\n## 输出要求\n\n* 输出修复后的完整 RST 正文（不加多余说明或包裹）。\n* 如需 diff 模式，由外层系统控制；本指令默认仅产出修复后全文。\n\n## 审核自查清单（模型内部执行，无需输出）\n\n* [ ] 指令语法/缩进/空行正确\n* [ ] 行内标记无空格、边界空格正确\n* [ ] 文件名/标题/API 名一致（含 `func_` 例外）\n* [ ] 参数/关键字参数：顺序与分流严格按签名（含 `/`、裸 `*`、`*args`、`**kwargs`）\n* [ ] “可选/默认值”标注与默认值文字格式正确\n* [ ] 异常：格式统一、同类聚合、无臆测\n* [ ] 输入/输出：列表格式统一\n* [ ] 列表缩进 2 空格，段落不缩进\n* [ ] 模块间有空行（例如返回是一个模块，异常是一个模块，他们之间有空行，但是异常下面的具体异常和异常大标题之间无空行）\n* [ ] 中文标点与顿号、错别字修复\n* [ ] 代码/公式内容未被改写（仅围栏/空格修复）"),
      maxTokens: config.get<number>('generation.maxTokens', 4096),
      temperature: config.get<number>('generation.temperature', 0.3),
      neverUploadIfWorkspaceTrusted: config.get<boolean>('safety.neverUploadIfWorkspaceTrusted', false),
      rewrapWidth: config.get<number>('format.rewrapWidth', 0)
    };
  }

  static validateConfig(config: OptimizationConfig): string[] {
    const errors: string[] = [];

    if (!config.baseUrl) {
      errors.push('API 基础 URL 不能为空');
    }

    if (!config.model) {
      errors.push('模型名称不能为空');
    }

    if (!config.apiKey) {
      errors.push('API 密钥不能为空，请在设置中配置或设置环境变量 RST_OPTIMIZER_API_KEY');
    }

    if (config.maxTokens <= 0) {
      errors.push('最大 token 数量必须大于 0');
    }

    if (config.temperature < 0 || config.temperature > 2) {
      errors.push('温度值必须在 0-2 之间');
    }

    return errors;
  }

  static checkWorkspaceSafety(): boolean {
    const workspaceTrust = vscode.workspace.isTrusted;
    const config = this.getConfig();
    
    if (config.neverUploadIfWorkspaceTrusted && workspaceTrust) {
      vscode.window.showWarningMessage(
        '安全设置阻止了在受信任工作区中上传内容到外部 API。请在设置中关闭 "neverUploadIfWorkspaceTrusted" 选项。'
      );
      return false;
    }

    return true;
  }
}