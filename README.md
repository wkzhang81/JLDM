# 支持多模态的快速类Jev决策机
## Jev-like Fast Decision Machine (JLDM) with image support

基于 [KoboldCpp](https://github.com/LostRuins/koboldcpp) 本地推理后端的 **Choice\<T\> 概率决策工具**。
输入一段上下文、**一张图片**、一个评估准则和若干候选选项，模型通过 GBNF Grammar 被严格约束为只输出 A/B/C… 字母，随后从该答题 token 的 logprobs 中提取各候选的对数概率，经温度校准后输出完整的后验概率分布。
常规决策速度介于300 ~ 600 ms。

同时提供：
- **Python CLI**（`jev_decision.py`）：支持批量 JSON 输入、无第三方依赖；
- **Web UI**（`index.html`）：单文件网页，双击即可在浏览器中使用，**支持拖拽图片做多模态决策**。

---

## 工作原理

```
用户输入 (reference + question + options)
        │
        ▼
  构造英文 Prompt（与 OpenAI /chat/completions 同构）
        │
        ▼
  KoboldCpp /api/v1/generate
  ├─ GBNF grammar: root ::= [ABC]   ← 词表硬约束为候选字母
  ├─ max_length = 1, temperature = 1.0
  └─ 关闭所有会截断 top-k 的采样器
        │
        ▼
  KoboldCpp /api/extra/last_logprobs
  → 拿到答题位上每个候选字母 token 的 logprob
        │
        ▼
  同字母变体 logsumexp 合并 → softmax(temperature)
        │
        ▼
  winner + 置信度 + 各选项概率百分比
```

核心思路：不依赖模型"自己说出"答案，而是在 token 层直接读取各候选的对数概率，再做条件归一化。这样得到的是**候选集合内的完整概率分布**，而非单次采样的单点结果。

---

## 前置条件

### 1. 下载 KoboldCpp

从 [KoboldCpp Releases](https://github.com/LostRuins/koboldcpp/releases) 下载对应平台的可执行文件（Windows 推荐 `koboldcpp_cu12.exe`）。

### 2. 下载模型文件

本项目默认配合以下 GGUF 量化模型使用（27B 量级，IQ3_XXS 量化 + MTP，支持视觉）：

| 文件 | 用途 | 下载地址 |
|------|------|----------|
| `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored-MTP.gguf` | 主模型权重 | [resolve 下载](https://huggingface.co/RentedNoodle/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored/resolve/main/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored-MTP.gguf) |
| `mmproj-Qwen3.8-27B-BF16.gguf` | 多模态投影（图片输入用，放在 `mmproj/` 子目录） | [resolve 下载](https://huggingface.co/RentedNoodle/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored/resolve/main/mmproj/mmproj-Qwen3.8-27B-BF16.gguf) |

> **注意**：HuggingFace 页面上的 `blob/main/...` 链接是预览页，实际下载需用 `resolve/main/...` 格式。也可以用 `huggingface-cli download` 或 `wget` / `curl` 直接拉取。

### 3. 启动 KoboldCpp

**GUI 方式（推荐新手）：**
1. 打开 `koboldcpp.exe`
2. **Model** 选择主模型 `.gguf` 文件
3. 勾选 **Use mmproj**，选择 `mmproj-Qwen3.8-27B-BF16.gguf`
4. **Port** 设为 `5001`
5. 点击 **Start**

**命令行方式：**

```bash
koboldcpp_cu12.exe \
  --model Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored-MTP.gguf \
  --mmproj mmproj/mmproj-Qwen3.8-27B-BF16.gguf \
  --port 5001 \
  --host 127.0.0.1 \
  --flashattention
```

启动后确认浏览器访问 `http://localhost:5001/v1/models` 能返回 JSON。

### 4. CORS 说明

Web UI 从浏览器直接请求 KoboldCpp，需要后端允许跨域。KoboldCpp 默认在 `127.0.0.1` 下通常可直接访问；若遇到 CORS 报错：
- 启动时加 `--host 0.0.0.0`（注意安全风险）；
- 或用浏览器插件允许跨域；
- 或直接用 Python CLI（无 CORS 问题）。

---

## 快速开始

### Web UI

直接用浏览器打开 `index.html`：

```bash
# 方式一：直接双击 index.html
# 方式二：本地起个静态服务（避免 file:// 限制）
python -m http.server 8080
# 然后访问 http://localhost:8080/index.html
```

界面操作：
1. 顶部确认连接地址为 `http://localhost:5001/v1`
2. 在左侧填写背景描述和评估准则
3. （可选）拖拽截图到图片区域做多模态判断
4. 在右侧添加候选选项（A/B/C/D…）
5. 点击「运行决策」，下方展示后验概率分布

### Python CLI

```bash
# 单条决策
python jev_decision.py --input examples/input_single.json

# 批量决策（11 条工单分类示例）
python jev_decision.py --input examples/input_batch_local.json --output result.json --debug

# 从 stdin 读取
echo '{"reference":"...","question":"...","options":["A","B","C"]}' | python jev_decision.py --input -
```

无第三方依赖，Python 3.8+ 标准库即可运行。

---

## 输入格式

CLI 接受三种 JSON 形态：

**单条对象：**
```json
{
  "reference": "凌晨3点收到陌生IP登录告警…",
  "question": "将工单分类到正确部门。",
  "options": ["账单计费", "技术支持", "安全事件", "人工审核"],
  "mode": "local",
  "base_url": "http://localhost:5001/v1",
  "temperature": 1.5
}
```

**批量数组：**
```json
[ { "reference": "...", "question": "...", "options": [...] }, ... ]
```

**带共享连接配置的批量：**
```json
{
  "mode": "local",
  "base_url": "http://localhost:5001/v1",
  "temperature": 1.5,
  "items": [
    { "id": "ticket-001", "reference": "...", "question": "...", "options": [...] },
    { "id": "ticket-002", "reference": "...", "question": "...", "options": [...] }
  ]
}
```

**字段说明：**

| 字段 | 必填 | 说明 |
|------|------|------|
| `reference` | 否 | 背景上下文，会被组装进 state JSON |
| `question` | 是 | 评估准则 / 问题描述 |
| `options` | 是 | 候选选项字符串数组，Local 模式最多 5 个 |
| `images` | 否 | 本地图片路径数组（多模态） |
| `mode` | 否 | `local`（KoboldCpp 原生接口）或 `cloud`（OpenAI 兼容 /chat/completions），默认 local |
| `base_url` | 是 | API 地址，本地默认 `http://localhost:5001/v1` |
| `model` | cloud 必填 | 云端模式的模型名 |
| `api_key` | 否 | 本地可留空 |
| `temperature` | 否 | 概率校准温度，默认 1.5。越高越平滑，越低越尖锐 |

## 输出格式

```json
{
  "winner": "安全事件",
  "confidence": 0.9651,
  "probabilities_pct": {
    "账单计费": 1.08,
    "技术支持": 1.44,
    "安全事件": 96.51,
    "人工审核": 0.97
  },
  "elapsed_ms": 457.2,
  "mode": "local",
  "image_count": 0,
  "answer_token_pos": 0,
  "used_fallback": false,
  "index": 1,
  "id": "ticket-002"
}
```

加 `--debug` 会额外输出完整的 logprobs 解析过程、grammar 和原始请求。

---

## 目录结构

```
jev-like/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── jev_decision.py          # Python CLI 决策脚本
├── index.html               # Web UI（单文件，双击即用）
└── examples/
    ├── input_batch_local.json    # 批量输入示例（含图片多模态）
    ├── output_batch_local.json   # 批量输出示例
    └── error.png                 # 多模态测试截图
```

---

## 已知限制

- **Local Grammar 模式最多 5 个候选**：KoboldCpp 1.121 的 `last_logprobs` 最多返回 5 个 top picks。超过 5 个选项请用 Cloud 模式。
- **Local 模式要求 KoboldCpp 版本 ≥ 1.121**，需支持 `/api/extra/last_logprobs` 和 GBNF grammar。
- 同字母的多个 token 变体（如 `"C"`、`" C"`、`"\nC"`）会在 log-space 做 logsumexp 合并。
- 未在答题位出现的候选不会被伪造为概率，会在 `missing_options` 中标注。

## License

MIT
