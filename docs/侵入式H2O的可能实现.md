这是一个非常棒的工程决策！既然队友使用了“侵入式”（Monkey Patching + 自定义 Cache 类）的方案，这实际上为你实现 H2O 算法打开了“上帝模式”。

**为什么？**
*   **Hook 的局限**：Hook 只能在 `forward` 之前（Pre）或之后（Post）运行。在 `Pre` 时，Attention 分数还没算出来；在 `Post` 时，显存已经占用了。要实现“算完 Attention -> 更新分数 -> 下一步根据分数驱逐”，Hook 的数据流很别扭。
*   **队友方案的优势**：既然我们直接重写了 `forward` 函数（Monkey Patch），我们就可以在计算完 Attention 的**那一瞬间**，直接把分数“喂回”给 Cache 对象。这就是一个完美的**反馈闭环（Feedback Loop）**。

### 核心思路：反馈闭环 (The Feedback Loop)

我们需要对队友的代码做三处关键修改，实现 H2O：

1.  **修改 Patch 函数（眼睛）**：在 `patched_gpt_neox_attention_forward` 里，计算完 Attention 后，不要扔掉 `attn_weights`，而是检查一下 Cache 对象：“嘿，你需要这个分数吗？”如果需要，就传给它。
2.  **实现 H2O Cache（大脑）**：继承队友的 `StreamingDynamicCache`，新增一个 `accumulated_scores` 字典。每次收到分数就累加；每次 Cache 满了，就根据“Sink + Recent + **High Score**”的规则来删人。
3.  **注册函数（入口）**：新增一个 `enable_h2o_llm`，用来启用这个新模式。

---

### 代码修改指南

请直接修改/追加以下内容到你的 `pythia_streaming_patch.py` 文件中。

#### 第一步：修改 Patch 函数 (建立反馈通道)

找到 `patched_gpt_neox_attention_forward` 函数，修改其中调用 `attention_interface` 后的部分：

```python
# 在 pythia_streaming_patch.py 中替换原有的 patched_gpt_neox_attention_forward

def patched_gpt_neox_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, 3 * self.head_size)

    qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states, value_states = qkv.chunk(3, dim=-1)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # 1. 更新 Cache (这里会触发 H2O 的驱逐逻辑)
    if layer_past is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "partial_rotation_size": self.rotary_ndims,
            "cache_position": cache_position,
        }
        key_states, value_states = layer_past.update(key_states, value_states, self.layer_idx, cache_kwargs) 

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # --- TIMING BLOCK ---
    start_t = None
    if TIMING_ENABLED and hidden_states.shape[1] == 1:
        torch.cuda.synchronize()
        start_t = time.time()
    # --------------------

    # 2. 计算 Attention (强制获取 attn_weights)
    # H2O 必须需要 attention weights，所以我们尽量传递 output_attentions=True
    # Eager 模式下通常默认返回 (output, weights)
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        head_mask=head_mask,
        output_attentions=True, # [H2O] 显式请求权重
        **kwargs,
    )

    # [H2O 核心逻辑] 3. 反馈闭环：把 Attention 分数传回给 Cache
    if layer_past is not None and hasattr(layer_past, "update_scores") and attn_weights is not None:
        layer_past.update_scores(attn_weights, self.layer_idx)

    # --- TIMING RECORD ---
    if start_t is not None:
        torch.cuda.synchronize()
        end_t = time.time()
        ATTENTION_TIMES.append((end_t - start_t) * 1000) 
    # ---------------------

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.dense(attn_output)

    return attn_output, attn_weights
```

#### 第二步：实现 H2O Cache 类 (大脑)

在 `StreamingDynamicCache` 类之后，添加这个新类：

```python
# 在 pythia_streaming_patch.py 中追加

class H2ODynamicCache(StreamingDynamicCache):
    """
    实现了 H2O (Heavy Hitter Oracle) 策略。
    保留: [Sinks] + [Heavy Hitters] + [Recent Window]
    """
    def __init__(self, config, n_sink=4, recent_window=32, max_capacity=256, debug=False):
        # 注意：这里传给父类的 window_size 实际上是我们的 max_capacity (总预算)
        # 我们会重写 update 方法，所以父类的驱逐逻辑不会被执行，但我们需要它的结构
        super().__init__(config, n_sink=n_sink, window_size=max_capacity, debug=debug)
        
        self.max_capacity = max_capacity
        self.recent_window = recent_window
        self.n_sink = n_sink
        self.heavy_hitter_size = max_capacity - n_sink - recent_window
        
        # 存储每层的累积注意力分数
        # Key: layer_idx, Value: Tensor [current_seq_len]
        self.accumulated_scores = {}

    def update_scores(self, attn_weights, layer_idx):
        """
        接收来自 Attention 层的分数并累加。
        attn_weights: [batch, heads, q_len, k_len]
        """
        if layer_idx not in self.accumulated_scores:
            return

        # 我们主要关注 KV 的重要性，也就是 dim=-1 (k_len)
        # 对 Batch, Heads, Queries 进行求和，得到每个 Key 被关注的总量
        # sum over (0, 1, 2) -> [k_len]
        # 注意：attn_weights 可能是 [1, 32, 1, 256]，sum 后变成 [256]
        
        # 稳健性处理：处理可能的维度差异
        dims_to_sum = [0, 1]
        if attn_weights.dim() == 4:
            dims_to_sum.append(2) # sum over q_len
            
        step_scores = attn_weights.sum(dim=dims_to_sum)
        
        # 确保数据在同一个设备上
        current_scores = self.accumulated_scores[layer_idx]
        
        # 对齐长度 (处理可能的边界情况)
        if step_scores.shape[0] == current_scores.shape[0]:
            self.accumulated_scores[layer_idx] += step_scores.detach()
        else:
            # 这种情况通常发生在 prefill 阶段之后或者维度不匹配
            # 简单起见，如果长度不匹配（且不是由于驱逐引起的），我们跳过更新或截断
            min_len = min(step_scores.shape[0], current_scores.shape[0])
            self.accumulated_scores[layer_idx][:min_len] += step_scores[:min_len].detach()

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        # 1. 记录长度
        if layer_idx not in self._seen_tokens_by_layer:
            self._seen_tokens_by_layer[layer_idx] = 0
        self._seen_tokens_by_layer[layer_idx] += key_states.shape[-2]
        
        # 2. 初始化分数存储 (如果是该层第一次运行)
        if layer_idx not in self.accumulated_scores:
            # 初始化为 0，长度为 0
            self.accumulated_scores[layer_idx] = torch.zeros(0, device=key_states.device)

        # 3. 扩展分数 Tensor (为新进来的 tokens 补 0)
        # key_states 是新进来的 token，长度通常为 1 (生成阶段)
        new_token_count = key_states.shape[-2]
        zeros = torch.zeros(new_token_count, device=key_states.device)
        self.accumulated_scores[layer_idx] = torch.cat([self.accumulated_scores[layer_idx], zeros], dim=0)

        # 4. 标准 Update (追加 KV)
        k_out, v_out = super(DynamicCache, self).update(key_states, value_states, layer_idx, cache_kwargs)
        
        # 5. 驱逐逻辑 (H2O 核心)
        current_len = k_out.shape[-2]
        
        # 使用 Lazy Eviction: 只有当超出 buffer (64) 时才进行昂贵的 TopK 操作
        if current_len > self.max_capacity + 64:
            
            # --- 确定要保留的索引 ---
            indices = torch.arange(current_len, device=k_out.device)
            scores = self.accumulated_scores[layer_idx]
            
            # A. Sink Mask (前 n_sink 个)
            sink_mask = indices < self.n_sink
            
            # B. Recent Window Mask (最后 recent_window 个)
            window_mask = indices >= (current_len - self.recent_window)
            
            # C. Candidates (中间的部分)
            # 我们只能从中间部分挑 Heavy Hitters
            candidate_mask = ~(sink_mask | window_mask)
            candidate_indices = indices[candidate_mask]
            candidate_scores = scores[candidate_mask]
            
            # D. Select Heavy Hitters
            # 选分数最高的 k 个
            num_hh = self.heavy_hitter_size
            if num_hh > 0 and len(candidate_scores) > 0:
                # TopK
                top_k = min(num_hh, len(candidate_scores))
                _, top_indices_local = torch.topk(candidate_scores, k=top_k)
                hh_indices = candidate_indices[top_indices_local]
            else:
                hh_indices = torch.tensor([], device=k_out.device, dtype=torch.long)
            
            # E. 合并所有保留的索引
            keep_indices = torch.cat([indices[sink_mask], hh_indices, indices[window_mask]])
            
            # F. 排序 (保持时间顺序非常重要！)
            keep_indices, _ = torch.sort(keep_indices)
            
            # --- 执行裁剪 ---
            
            # 1. 裁剪 KV Cache
            k_new = k_out[:, :, keep_indices, :]
            v_new = v_out[:, :, keep_indices, :]
            
            if layer_idx < len(self.layers):
                self.layers[layer_idx].keys = k_new
                self.layers[layer_idx].values = v_new
            
            # 2. 同步裁剪 分数 Tensor (必须一一对应)
            self.accumulated_scores[layer_idx] = scores[keep_indices]
            
            return k_new, v_new

        return k_out, v_out
        
    def evict_all_layers(self):
        # PPL 测试时手动触发驱逐，逻辑同上，为了简洁这里略去
        # 如果需要跑 PPL，建议在 evaluate_ppl_unified 里调用 model(..., output_attentions=True)
        pass
```

#### 第三步：注册函数

添加一个启用函数：

```python
# 在 pythia_streaming_patch.py 中追加

def enable_h2o_llm(model: GPTNeoXForCausalLM, n_sink=4, recent_window=32, max_capacity=256, debug=False):
    """启用 H2O 模式"""
    # 存储配置
    model._h2o_config = (n_sink, recent_window, max_capacity, debug)
    
    # 如果还没有 patch 过 forward，先 patch
    if not hasattr(model, "_original_forward_streaming_patch"):
        model._original_forward_streaming_patch = model.forward

        def streaming_forward_wrapper(self, input_ids=None, past_key_values=None, use_cache=None, **kwargs):
            # 获取配置
            if hasattr(self, "_h2o_config"):
                n_sink, recent_window, max_capacity, debug = self._h2o_config
                CacheClass = H2ODynamicCache
                cache_args = {
                    "n_sink": n_sink, 
                    "recent_window": recent_window, 
                    "max_capacity": max_capacity, 
                    "debug": debug
                }
            else:
                # 回退到 StreamingLLM 配置
                n_sink, window_size, debug = self._streaming_config
                CacheClass = StreamingDynamicCache
                cache_args = {"n_sink": n_sink, "window_size": window_size, "debug": debug}

            # 注入 Cache
            if use_cache:
                if past_key_values is None:
                    past_key_values = CacheClass(self.config, **cache_args)
                elif isinstance(past_key_values, DynamicCache) and not isinstance(past_key_values, CacheClass):
                    if past_key_values.get_seq_length() == 0:
                         past_key_values = CacheClass(self.config, **cache_args)
            
            return self._original_forward_streaming_patch(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs
            )
        
        model.forward = types.MethodType(streaming_forward_wrapper, model)
    
    # 确保 Attention 层被 patch 过了
    patch_attention_layers(model)
```

---

### 如何在 `main.py` 中使用？

修改你的 `configs` 列表，加入 H2O 的配置：

```python
from pythia_streaming_patch import enable_h2o_llm

configs = [
    # ... 之前的配置 ...
    # max_capacity=256: 4个Sink + 32个最近 + 220个Heavy Hitters
    {"name": "h2o_256", "type": "h2o", "sink": 4, "window": 32, "capacity": 256},
]

# 在 run_standard_benchmark 循环里：
    for config in configs:
        # ...
        if config["type"] == "baseline":
            # ...
        elif config["type"] == "streaming":
            # ...
        elif config["type"] == "h2o":
             enable_h2o_llm(
                model, 
                n_sink=config["sink"], 
                recent_window=config["window"], 
                max_capacity=config["capacity"], 
                debug=False
            )
```

**这样，你就成功地在队友的“侵入式”架构上，实现了一个带有反馈闭环的高级 H2O 算法！** 这种组合（Monkey Patch 核心循环 + Feedback Loop + Lazy Eviction）是一个非常有深度的工程实现。