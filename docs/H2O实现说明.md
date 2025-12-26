# H2O 实现说明

## 文件结构

### 新创建的文件
1. **`pythia_streaming_h2o_patch.py`** - H2O 核心实现
   - 从 `pythia_streaming_patch.py` 复制并修改
   - 实现了 H2O (Heavy Hitter Oracle) 算法

2. **`bench_streaming_h2o.py`** - H2O 测试脚本
   - 从 `main.py` 复制并修改
   - 添加了 H2O 配置和测试逻辑

### 保留的原始文件
- `pythia_streaming_patch.py` - 原始 StreamingLLM 实现（不变）
- `main.py` - 原始测试脚本（不变）

## 核心实现

### 1. H2ODynamicCache 类

保留策略：**[Sinks] + [Heavy Hitters] + [Recent Window]**

```
配置示例：n_sink=4, recent_window=32, max_capacity=256
结构：[4个Sink] + [220个Heavy Hitters] + [32个Recent]
```

**关键特性：**
- **Sink Tokens**：保留最初的 4/8 个 tokens（位置编码重要性）
- **Heavy Hitters**：根据累积注意力分数选择最重要的 tokens
- **Recent Window**：保留最近的 32/64 个 tokens（局部性原理）
- **Lazy Eviction**：超过 `max_capacity + 64` 才触发驱逐，避免频繁操作

### 2. 反馈闭环 (Feedback Loop)

```python
# 在 patched_gpt_neox_attention_forward 中：
force_output_attentions = output_attentions or (layer_past is not None and hasattr(layer_past, "update_scores"))

# 计算 Attention 后立即反馈
if layer_past is not None and hasattr(layer_past, "update_scores") and attn_weights is not None:
    layer_past.update_scores(attn_weights, self.layer_idx)
```

**工作流程：**
1. Attention 计算完成
2. 获取 `attn_weights` [batch, heads, q_len, k_len]
3. 对 batch, heads, queries 维度求和 → [k_len]
4. 累加到 `accumulated_scores[layer_idx]`
5. 驱逐时根据分数选择 Top-K 作为 Heavy Hitters

### 3. 分数累积策略

```python
def update_scores(self, attn_weights, layer_idx):
    # 对 Batch, Heads, Queries 求和
    dims_to_sum = [0, 1, 2]
    step_scores = attn_weights.sum(dim=dims_to_sum)  # [k_len]
    
    # 累加到历史分数
    self.accumulated_scores[layer_idx] += step_scores.detach()
```

## 测试配置

### 配置列表（bench_streaming_h2o.py）

```python
configs = [
    # Baseline：无优化
    {"name": "baseline", "type": "baseline"},
    
    # StreamingLLM：原始方法
    {"name": "streaming_8_256", "type": "streaming", "sink": 8, "window": 256},
    {"name": "streaming_8_512", "type": "streaming", "sink": 8, "window": 512},
    
    # H2O：改进方法
    {"name": "h2o_4_32_256", "type": "h2o", "sink": 4, "recent": 32, "capacity": 256},
    {"name": "h2o_8_32_256", "type": "h2o", "sink": 8, "recent": 32, "capacity": 256},
    {"name": "h2o_8_64_512", "type": "h2o", "sink": 8, "recent": 64, "capacity": 512},
]
```

### 配置对比

| 配置            | Sink | Middle | Recent | 总容量 | 说明              |
| --------------- | ---- | ------ | ------ | ------ | ----------------- |
| streaming_8_256 | 8    | -      | 248    | 256    | StreamingLLM 基准 |
| h2o_4_32_256    | 4    | 220 HH | 32     | 256    | 激进 H2O          |
| h2o_8_32_256    | 8    | 216 HH | 32     | 256    | 平衡 H2O          |
| h2o_8_64_512    | 8    | 440 HH | 64     | 512    | 宽松 H2O          |

## 运行方法

### 运行 H2O 测试
```bash
python bench_streaming_h2o.py
```

### 运行原始 StreamingLLM（对照组）
```bash
python main.py
```

### 调试模式
```bash
python bench_streaming_h2o.py test
```

## 预期结果

### 性能预期

**相比 StreamingLLM (streaming_8_256):**
- ✅ **PPL 下降**：应该从 32.24 降低到 15-25 左右
- ✅ **显存相同**：保持 ~5.31 GB
- ⚠️ **速度略降**：TopK 操作会增加约 5-10% 开销
- ✅ **Throughput 相近**：应该保持在 26-28 tok/s

**H2O vs StreamingLLM 的核心优势：**
在相同显存预算下，H2O 通过智能选择重要 tokens，能够更好地保留长距离依赖关系，从而显著降低 PPL。

### 评估指标

关键对比：
1. **PPL (越低越好)**：主要指标，衡量模型质量
2. **Peak Mem**：应该相同
3. **Throughput**：H2O 略低（TopK 开销）
4. **Avg Attn (ms)**：H2O 略高（权重计算）

## 理论依据

### H2O 核心思想

论文：[H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)

**关键洞察：**
1. **注意力分布不均**：少数 tokens 累积了大部分注意力权重
2. **Heavy Hitter Tokens**：这些高权重 tokens 对模型输出至关重要
3. **动态选择**：通过累积注意力分数动态识别重要 tokens

**与 StreamingLLM 的区别：**
- StreamingLLM：固定保留 [Sink + Recent]，简单但粗糙
- H2O：动态保留 [Sink + Heavy Hitters + Recent]，智能且精准

## 代码亮点

### 1. 非侵入式设计
- 完全继承 `StreamingDynamicCache`
- 通过 Monkey Patching 实现，无需修改 Transformers 源码

### 2. Lazy Eviction
```python
if current_len > self.max_capacity + 64:  # Buffer 64
    # 执行昂贵的 TopK 操作
```
避免每一步都做驱逐，提升效率

### 3. 时间顺序保持
```python
keep_indices, _ = torch.sort(keep_indices)
```
保持 tokens 的时间顺序对位置编码至关重要

### 4. 分数同步裁剪
```python
# KV Cache 和分数必须一一对应
self.layers[layer_idx].keys = k_new
self.accumulated_scores[layer_idx] = scores[keep_indices]
```

## 下一步

1. **运行测试**：`python bench_streaming_h2o.py`
2. **对比结果**：与 `python main.py` 的结果对比
3. **调优参数**：
   - 调整 `heavy_hitter_size`（220 vs 440）
   - 调整 `recent_window`（32 vs 64）
   - 调整 `max_capacity`（256 vs 512）
4. **分析报告**：记录到 GitHub Issue #8

## 预期实验报告内容

对比表格示例：

| Configuration    | Wikitext PPL | Δ PPL vs Baseline | Peak Mem (GB) | Throughput (tok/s) |
| ---------------- | ------------ | ----------------- | ------------- | ------------------ |
| baseline         | 6.99         | -                 | 5.48          | 26.92              |
| streaming_8_256  | **32.24**    | +361%             | 5.31          | 28.57              |
| **h2o_8_32_256** | **~18.00**   | +157%             | 5.31          | 27.50              |

**结论预期：**
H2O 在相同显存条件下，将 PPL 下降约 **44%**（从 32.24 → 18.00），显著改善了模型质量。
