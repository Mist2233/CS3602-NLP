# NLP Final Lab - StreamingLLM Implementation

基于 Pythia-2.8b 模型的 StreamingLLM KV Cache 优化实验

## 📋 目录

- [项目简介](#项目简介)
- [环境配置](#环境配置)
- [模型与数据集下载](#模型与数据集下载)
- [文件说明](#文件说明)
- [运行方法](#运行方法)
- [实验结果](#实验结果)
- [FAQ](#faq)

---

## 🎯 项目简介

本项目实现了 **StreamingLLM** 算法，通过智能压缩 KV Cache 来优化大语言模型的推理性能。主要特点：

- ✅ **质量无损**: PPL 保持不变 (9.79 → 9.79)
- ✅ **内存优化**: 显存占用降低 10.0% (6100 MB → 5493 MB)
- ✅ **性能提升**: 吞吐量提升 33.9% (24.40 → 32.68 tokens/s)
- ✅ **延迟降低**: TTFT 降低 45.9%, TPOT 降低 25.2%
- ✅ **完整实现**: 使用 Pre-Forward Hook 正确拦截和修改 DynamicCache

**核心思想**: 保留开头的 Attention Sinks (n_sink tokens) 和末尾的最近 tokens，丢弃中间的过时 tokens。

---

## 🔧 环境配置

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

## 📦 模型与数据集下载

### 方法一：自动下载（推荐）

运行下载脚本会自动从 HuggingFace Mirror 下载：

```bash
conda activate nlp
python download_model.py
```

下载内容：
- **模型**: Pythia-2.8b (EleutherAI/pythia-2.8b)
- **保存位置**: `./models/pythia-2.8b/`
- **模型大小**: ~5 GB
- **预计下载时间**: 5-20 分钟（取决于网速）

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

## 📁 文件说明

### 核心脚本

| 文件                     | 说明                  | 用途                                         |
| ------------------------ | --------------------- | -------------------------------------------- |
| `download_model.py`      | 模型下载脚本          | 从 HuggingFace 下载 Pythia-2.8b              |
| `baseline.py`            | 基准测试脚本          | 测试原始模型的 PPL、Memory、FLOPs 等指标     |
| `benchmark_streaming.py` | StreamingLLM 对比测试 | 对比 Baseline 和 StreamingLLM 的全部性能指标 |
| `pythia_press.py`        | StreamingLLM 核心实现 | KV Cache 压缩器，使用 Pre-Forward Hook       |
| `run_pythia.py`          | 简单推理脚本          | 快速测试模型生成能力                         |

### 调试与文档

| 文件                     | 说明                 |
| ------------------------ | -------------------- |
| `debug_press.py`         | 详细调试工具（已归档） | 验证压缩逻辑，对比三种模式 |
| `streaming_llm_press.py` | 旧版实现（已废弃）   | 历史版本，不推荐使用       |
| `FIX_SUMMARY.md`         | 修复总结文档         | 详细记录调试过程和解决方案 |
| `worklog.md`             | 工作日志             | 开发过程记录               |
| `README.md`              | 本文件               | 项目说明文档               |

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
├── baseline.py                # 基准测试
├── benchmark_streaming.py     # StreamingLLM 对比
├── pythia_press.py           # 核心实现
├── download_model.py         # 下载脚本
└── README.md                 # 本文件
```

---

## 🚀 运行方法

### 1. 下载模型（首次运行必需）

```bash
conda activate nlp
python download_model.py
```

预计下载时间：20-40 分钟（取决于网速）
模型大小：约 10 GB

### 2. 快速测试生成效果

```bash
python run_pythia.py
```

这会快速生成一段文本，验证模型加载正确。

### 3. 基准测试（Baseline）

测试原始模型性能：

```bash
python baseline.py
```

输出指标：
- **PPL**: 困惑度（WikiText-2 和 PG-19）
- **Memory**: 峰值显存占用
- **FLOPs**: 模型计算量
- **Speed**: 吞吐量、TTFT、TPOT

预计运行时间：~5 分钟

### 4. StreamingLLM 对比测试（核心实验）

```bash
python benchmark_streaming.py
```

这会运行：
1. Baseline 测试
2. StreamingLLM 测试（compression_ratio=0.7, n_sink=4）
3. 对比两者的性能差异

输出对比表格：
```
指标              | Baseline     | StreamingLLM | 变化
-------------------------------------------------------
PPL              | 9.79         | 9.79         | +0.0%
Memory (MB)      | 6100.46      | 5493.35      | -10.0%
Throughput (t/s) | 24.40        | 32.68        | +33.9%
TTFT (s)         | 0.3185       | 0.1724       | -45.9%
TPOT (ms)        | 40.84        | 30.53        | -25.2%
```


### 5. 调试工具（可选）

如果需要详细验证压缩逻辑：

```bash
python debug_press.py
```

输出包括：
- 每一步的 KV Cache 长度
- 压缩前后的验证
- 三种模式的对比（Baseline / Manual / Generate）

---

## 📊 实验结果

### 最终性能对比

| 指标               | Baseline   | StreamingLLM | 变化          | 说明              |
| ------------------ | ---------- | ------------ | ------------- | ----------------- |
| **PPL** (↓)        | 9.79       | 9.79         | **+0.0%** ✅   | 质量无损          |
| **Memory** (↓)     | 6100.46 MB | 5493.35 MB   | **-10.0%** ✅  | 显存节省 607 MB   |
| **Throughput** (↑) | 24.40 t/s  | 32.68 t/s    | **+33.9%** ✅✅ | 吞吐量大幅提升    |
| **TTFT** (↓)       | 318.5 ms   | 172.4 ms     | **-45.9%** ✅  | 首 Token 加速     |
| **TPOT** (↓)       | 40.84 ms   | 30.53 ms     | **-25.2%** ✅  | 每 Token 延迟降低 |

### 关键发现

1. **质量保证**: PPL 保持完全一致 (9.79)，证明 Attention Sinks 策略有效
2. **内存优化**: 显存节省 607 MB (-10.0%)，对长序列效果更显著
3. **性能提升**: 吞吐量提升 33.9%，压缩反而加速了推理！
4. **延迟优化**: TTFT 降低 45.9%，TPOT 降低 25.2%，用户体验大幅改善
5. **全面胜出**: 相比小模型，2.8B 展现出 StreamingLLM 的真正优势

### StreamingLLM 参数说明

```python
press = PythiaStreamingLLMPress(
    compression_ratio=0.7,  # 压缩率：丢弃 70% 的中间 tokens
    n_sink=4                # 保留开头 4 个 Attention Sink tokens
)
```

参数调优建议：
- **compression_ratio**: 0.5-0.8 之间效果较好
- **n_sink**: 2-8 之间，太少影响质量，太多压缩效果差

### 计算量分析

```
Model: Pythia-2.8b
Params: 2.78 B (约为 70M 的 40 倍)
Memory: ~5.5 GB (FP16)
Layers: 32 个 Transformer 层
```

---

## ❓ FAQ

### Q1: 安装 PyTorch 时遇到 CUDA 版本不匹配

**问题**: CUDA 版本错误

**解决方案**:
可以参考：[菜鸟教程](https://www.runoob.com/pytorch/pytorch-install.html)安装合适版本的PyTorch。

```bash
# 检查 CUDA 版本
nvidia-smi

# 根据 CUDA 版本安装 PyTorch
# CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Q2: 下载模型时报错 `Connection timeout`

**问题**: 国内网络无法直接访问 HuggingFace

**解决方案**:
1. 确保设置了镜像环境变量：
   ```python
   os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
   ```
2. 或使用代理：
   ```bash
   export http_proxy=http://127.0.0.1:7890
   export https_proxy=http://127.0.0.1:7890
   ```

### Q3: PG-19 数据集加载失败

**问题**: `RuntimeError: Dataset scripts are no longer supported`

**原因**: datasets 库版本过高（3.x or 4.x）

**解决方案**:
```bash
pip install datasets==2.21.0
```

或者在代码中使用：
```python
load_dataset("pg19", split="train", streaming=True, trust_remote_code=False)
```

### Q4: 显存不足 `CUDA out of memory`

**问题**: GPU 显存不够运行模型

**最低要求**: Pythia-2.8b 需要至少 **6 GB** 显存（推荐 8 GB+）

**解决方案**:
1. 使用 StreamingLLM 压缩（可节省约 600 MB）
2. 减少测试序列长度（修改 `MAX_LENGTH`）
3. 使用更激进的压缩参数：
   ```python
   press = PythiaStreamingLLMPress(compression_ratio=0.8, n_sink=2)
   ```
4. 如果显存 < 6 GB，考虑使用 Pythia-1b 或 Pythia-410m

### Q5: StreamingLLM 没有压缩效果

**问题**: 显存占用没有明显下降

**可能原因**:
1. **序列太短**: StreamingLLM 在长序列（1000+ tokens）时效果才明显
2. **参数设置**: compression_ratio 太低或 n_sink 太大
3. **实现错误**: 确保使用的是 `pythia_press.py`，不是 `streaming_llm_press.py`

**验证方法**:
```bash
python debug_press.py
```
查看输出中的 "KV Cache 长度" 是否稳定维持在压缩后的大小。

### Q6: 如何在自己的代码中使用 StreamingLLM？

```python
from pythia_press import PythiaStreamingLLMPress
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    'models/pythia-2.8b',
    torch_dtype=torch.float16,
    device_map='cuda'
)
tokenizer = AutoTokenizer.from_pretrained('models/pythia-2.8b')

# 注册 StreamingLLM
press = PythiaStreamingLLMPress(compression_ratio=0.7, n_sink=4)
press.register(model)

# 正常使用 generate()
inputs = tokenizer("Hello", return_tensors='pt').to('cuda')
outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)

# 查看压缩次数
print(f"压缩次数: {press.compression_count}")

# 记得在结束时移除 hook
press.remove()
```

### Q7: 为什么 TPOT 和 Throughput 略有下降？

**原因**: StreamingLLM 需要在每次 forward 前执行压缩操作，会引入少量计算开销。

**这是正常的**: 论文中也观察到类似现象，这是用少量计算换取内存节省的合理权衡。

**优化建议**:
- 如果主要关注吞吐量，可以降低 compression_ratio（如 0.5）
- 如果主要关注内存，可以提高 compression_ratio（如 0.8）

### Q8: 如何调整压缩参数以获得更好效果？

**参数组合建议**:

| 场景             | compression_ratio | n_sink | 说明                   |
| ---------------- | ----------------- | ------ | ---------------------- |
| 最大内存节省     | 0.8               | 2      | 激进压缩，可能影响质量 |
| 平衡方案（推荐） | 0.7               | 4      | 本实验使用的配置       |
| 保守方案         | 0.5               | 6      | 质量最优，压缩效果一般 |

修改 `benchmark_streaming.py` 中的参数：
```python
# 第 9 行附近
COMPRESSION_RATIO = 0.7  # 调整这个值
# 第 174 行附近
press = PythiaStreamingLLMPress(
    compression_ratio=COMPRESSION_RATIO,
    n_sink=4  # 调整这个值
)
```

---

## 🔗 参考资料

- [StreamingLLM 论文](https://arxiv.org/abs/2309.17453)
- [Pythia 模型](https://github.com/EleutherAI/pythia)
- [FIX_SUMMARY.md](FIX_SUMMARY.md) - 详细的实现和调试过程

---

## 📧 联系方式

如有问题，请提交 Issue 或联系项目维护者。

**最后更新**: 2025-12-14



