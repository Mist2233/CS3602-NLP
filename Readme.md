# NLP Final Lab - StreamingLLM Implementation

基于 Pythia-2.8b 模型的 StreamingLLM KV Cache 优化实验

## 📋 目录

- [项目简介](#项目简介)
- [环境配置](#环境配置)
- [模型与数据集下载](#模型与数据集下载)
- [文件说明](#文件说明)
- [运行方法](#运行方法)
- [实验结果](#实验结果)

---

## 项目简介

本项目实现了 **StreamingLLM** 算法，通过智能压缩 KV Cache 来优化大语言模型的推理性能。主要特点：

- ✅ **质量可控**: PPL 适度上升 (6.98 → 11.60, +66.2%)，在可接受范围内
- ✅ **内存优化**: 显存占用降低 5.9% (5783 MB → 5441 MB)
- ✅ **性能提升**: 吞吐量提升 16.3% (25.68 → 29.87 tokens/s)
- ✅ **延迟降低**: TTFT 降低 35.8%, TPOT 降低 13.9%
- ✅ **正确实现**: 使用 Monkey Patch + 自定义 Cache 类实现 StreamingLLM

**核心思想**: 保留开头的 Attention Sinks (n_sink tokens) 和末尾的最近 tokens，丢弃中间的过时 tokens。

**实现方式**: 通过 Monkey Patch 替换模型的 forward 方法，注入自定义的 `StreamingDynamicCache` 类，在 cache 容量超出限制时自动执行驱逐策略。

**配置说明**: 本实验使用 Sink=8, Window=248 (总容量=256)，在 1000 token 的测试中取得了性能与质量的良好平衡。

### 实现方案探索

本项目在实现过程中尝试了两种方案：

#### 方案一：Pre-Forward Hook（失败）
**实现思路**: 使用 `register_forward_pre_hook` 拦截 Attention 层的输入，在每次 forward 前检查并压缩 KV Cache。

**失败原因**:
1. **过度压缩**: Hook 在每个 token、每层都触发，导致 87,712 次压缩（正常应该只在 cache 超限时压缩）
2. **时序错误**: Pre-forward hook 无法阻止 cache 增长，压缩后立即又被新 token 扩展
3. **层间不一致**: 每层独立压缩，破坏了多层之间的 cache 一致性
4. **PPL 暴增**: 导致 PPL 从 6.98 暴增到 133.50 (+1812%)，完全不可用

**教训**: StreamingLLM 的正确实现必须在 Cache 类内部进行，而不是通过外部 Hook 拦截。

#### 方案二：侵入式修改 + 自定义 Cache（成功）
**实现思路**: 
1. 创建 `StreamingDynamicCache` 类继承自 `DynamicCache`
2. 重写 `update()` 方法，在其中实现 Lazy Eviction 逻辑
3. 通过侵入式修改替换模型的 forward 方法，注入自定义 Cache
4. Cache 只在超出容量时触发压缩，而非每次 forward

**成功原因**:
1. **正确的压缩时机**: Cache 在内部判断是否超限，只在必要时压缩
2. **保持一致性**: 所有层使用同一个 Cache 对象，状态统一
3. **性能高效**: 使用 Lazy Eviction，添加 64 token 的 buffer 避免频繁压缩
4. **结果合理**: PPL 只增加 66.2%，显存和速度都有改善

**实现细节**:
```python
class StreamingDynamicCache(DynamicCache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # 1. 调用父类添加新 token
        k_out, v_out = super().update(...)
        
        # 2. 检查是否超出容量 (Lazy Eviction)
        if current_len > limit + 64:  # buffer=64
            # 3. 保留 [Sink + Window]
            k_new = torch.cat([k_sink, k_window], dim=-2)
            v_new = torch.cat([v_sink, v_window], dim=-2)
            
            # 4. 更新 cache
            self.layers[layer_idx].keys = k_new
            self.layers[layer_idx].values = v_new
        
        return k_out, v_out
```

---

## 环境配置

### 1. 创建 Conda 环境

```bash
# 创建名为 nlp 的 Python 3.10 环境
conda create -n nlp python=3.10 -y
conda activate nlp
```

### 2. 安装依赖

```bash
# PyTorch (CUDA 11.8 版本，根据你的 CUDA 版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Transformers 和相关库
pip install transformers datasets accelerate
pip install huggingface_hub

# 性能分析工具
pip install calflops

# 其他工具
pip install tqdm
```

### 3. 依赖版本说明

推荐版本：
- Python: 3.10+
- PyTorch: 2.0+
- Transformers: 4.35+
- datasets: 2.x (注意：3.x 及以上的版本可能导致 PG-19 数据集加载失败)
- CUDA: 11.8 或 12.1

---

## 模型与数据集下载

### 方法一：自动下载（推荐）

**下载模型**：
```bash
conda activate nlp
python download_model.py
```

下载内容：
- **模型**: Pythia-2.8b (EleutherAI/pythia-2.8b)
- **保存位置**: `./models/pythia-2.8b/`
- **模型大小**: ~5 GB
- **预计下载时间**: 5-20 分钟（取决于网速）

**下载数据集**：
```bash
python download_datasets.py
```

下载内容：
- **WikiText-2**: 用于 PPL 评估
- **PG-19 样本**: 用于生成速度测试
- **保存位置**: `./hf_cache/datasets/`
- **数据集大小**: ~50 MB
- **预计下载时间**: 1-5 分钟

### 方法二：手动配置

1. **设置 HuggingFace 镜像**（大陆用户必需）:
   ```python
   import os
   os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
   ```

2. **数据集自动下载**：
   - WikiText-2: 运行脚本时自动下载到 `./hf_cache/datasets/wikitext/`
   - PG-19: 运行脚本时自动下载到 `./hf_cache/datasets/pg19/`

---

## 文件说明

### 核心脚本

| 文件                            | 说明                  | 用途                                         |
| ------------------------------- | --------------------- | -------------------------------------------- |
| `download_model.py`             | 模型下载脚本          | 从 HuggingFace 下载 Pythia-2.8b              |
| `download_datasets.py`          | 数据集下载脚本        | 下载 WikiText-2 和 PG-19 到本地              |
| `benchmark_streaming.py`        | StreamingLLM 对比测试 | 对比 Baseline 和 StreamingLLM 的全部性能指标 |
| `pythia_streaming_press.py` | StreamingLLM 核心实现 | Monkey Patch + 自定义 Cache 类实现           |
| `run_pythia.py`                 | 简单推理脚本          | 快速测试模型生成能力                         |

### 实现探索（教学价值）

| 文件              | 说明                             | 状态                             |
| ----------------- | -------------------------------- | -------------------------------- |
| `pythia_press.py` | Hook方式实现（失败案例）         |  废弃，已经删除                           |
| 说明              | 使用Pre-Forward Hook导致过度压缩 | 教训：不能用Hook实现StreamingLLM |

### 文档

| 文件         | 说明         |
| ------------ | ------------ |
| `README.md`  | 项目说明文档 |
| `RESULT.md`  | 实验结果输出 |

### 目录结构

```
NLP-FinalLab/
├── models/                    # 模型文件
│   └── pythia-2.8b/
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer.json
├── hf_cache/                  # HuggingFace 缓存
│   ├── datasets/              # 数据集缓存
│   └── hub/                   # 模型缓存
│   └── modules/               # 功能模组
├── benchmark_streaming.py     # StreamingLLM 对比
├── pythia_streaming_press.py  # 核心实现
├── download_model.py          # 下载模型脚本
├── download_datasets.py       # 下载数据集脚本
└── README.md                  # 本文件
```

---

## 运行方法

### 1. 下载模型与数据集（只需首次运行时完成）

```bash
conda activate nlp
python download_model.py
python download_datasets.py
```

预计下载时间：20-40 分钟（取决于网速）
模型大小：约 10 GB

### 2. 快速测试模型生成效果

```bash
python run_pythia.py
```

这会快速生成一段文本，验证模型加载正确。

### 3. StreamingLLM 对比测试（核心实验）

```bash
python benchmark_streaming.py
```

这会运行：
1. Baseline 测试（全量 KV Cache）
2. StreamingLLM 测试（Sink=8, Window=248）
3. 对比两者的性能差异

输出对比表格：
```
Metric                         | Baseline     | Streaming    | Change
------------------------------------------------------------------------
Perplexity                     | 6.9805       | 11.6016      | ↑ 66.2%
Peak Memory (MB)               | 5783.06      | 5441.05      | ↓ 5.9%
Throughput (tok/s)             | 25.68        | 29.87        | ↑ 16.3%
Time to First Token (s)        | 0.2698       | 0.1733       | ↓ 35.8%
Time per Output Token (ms)     | 38.71        | 33.34        | ↓ 13.9%
Avg Attention Time (ms)        | 0.15         | 0.08         | ↓ 42.4%
```

预计运行时间：~10 分钟

---
- 每一步的 KV Cache 长度
- 压缩前后的验证
- 三种模式的对比（Baseline / Manual / Generate）

---

## 实验结果

### 最终性能对比

| 指标               | Baseline  | StreamingLLM | 变化         | 说明                 |
| ------------------ | --------- | ------------ | ------------ | -------------------- |
| **PPL** (↓)        | 6.98      | 11.60        | **+66.2%** ✅ | 质量略微下降，可接受 |
| **Memory** (↓)     | 5783 MB   | 5441 MB      | **-5.9%** ✅  | 显存节省 342 MB      |
| **Throughput** (↑) | 25.68 t/s | 29.87 t/s    | **+16.3%** ✅ | 吞吐量提升           |
| **TTFT** (↓)       | 269.8 ms  | 173.3 ms     | **-35.8%** ✅ | 首 Token 加速        |
| **TPOT** (↓)       | 38.71 ms  | 33.34 ms     | **-13.9%** ✅ | 每 Token 延迟降低    |
| **Avg Attn** (↓)   | 0.15 ms   | 0.08 ms      | **-42.4%** ✅ | Attention 效率提升   |

### 关键发现

1. ✅ **质量可控**: PPL 只增加 66.2%，远低于 Hook 方案的 1812%
2. ✅ **内存优化**: 显存节省 342 MB (5.9%)，长序列效果更显著
3. ✅ **性能提升**: 吞吐量提升 16.3%，延迟降低 13.9-42.4%
4. ✅ **实现正确**: 侵入式修改 + 自定义 Cache 方式完全正确
5. ✅ **参数合理**: Sink=8, Window=248 (总容量=256) 在 1000 token 测试中取得良好平衡

### 实现方案对比

| 方案         | PPL 增幅 | 压缩次数 | 结果   | 原因                        |
| ------------ | -------- | -------- | ------ | --------------------------- |
| Hook 方式    | +1812%   | 87,712   | ❌ 失败 | 每 token 每层都压缩，过度   |
| 侵入式修改 | +66.2%   | ~750     | ✅ 成功 | 只在 cache 超限时压缩，正确 |

**教训**: StreamingLLM 必须在 Cache 类内部实现驱逐逻辑，不能通过外部 Hook 拦截。

### StreamingLLM 参数说明

```python
# 在 benchmark_streaming.py 中配置
SINK_SIZE = 8      # Attention Sink 保留的初始 token 数量
WINDOW_SIZE = 248  # 滑动窗口大小
# 总容量 = SINK_SIZE + WINDOW_SIZE = 256
```

参数调优建议：
- **总容量 (Sink + Window)**:
  - 256: 当前配置，PPL +66.2%，性能提升明显
  - 512: 预计 PPL +20-30%，更平衡的选择
  - 1024: 预计 PPL +5-10%，质量接近 baseline
- **Sink 大小**: 4-8 之间，保留初始上下文的锚点
- **Window 大小**: 决定了最近历史的保留量，是主要参数

### 使用方法

```python
from pythia_streaming_press import enable_streaming_llm, disable_streaming_llm

# 加载模型
model = AutoModelForCausalLM.from_pretrained("./models/pythia-2.8b", ...)

# 启用 StreamingLLM
enable_streaming_llm(model, n_sink=8, window_size=248)

# 正常使用 generate()
outputs = model.generate(**inputs, max_new_tokens=1000, use_cache=True)

# 禁用 StreamingLLM（如需切换回 baseline）
disable_streaming_llm(model)
```

### 计算量分析

```
Model: Pythia-2.8b
Params: 2.78 B (约为 70M 的 40 倍)
Memory: ~5.5 GB (FP16)
Layers: 32 个 Transformer 层
```


## 参考资料

- [StreamingLLM 论文](https://arxiv.org/abs/2309.17453) - Efficient Streaming Language Models with Attention Sinks
- [Pythia 模型](https://github.com/EleutherAI/pythia) - EleutherAI's Suite of Models
- [Transformers DynamicCache](https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) - 官方文档

---

## 📧 联系方式

如有问题，请提交 Issue 或联系项目维护者。

**最后更新**: 2025-12-27



