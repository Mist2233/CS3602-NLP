import torch


class PythiaStreamingLLMPress:
    """
    针对 Pythia (GPT-NeoX) 架构复现的 StreamingLLM 压缩器。

    使用 Pre-Forward Hook 拦截 DynamicCache 并压缩 KV Cache。

    关键发现:
    - DynamicCache 访问: cache[layer_idx] 返回 (key, value)
    - 修改方式: cache.layers[layer_idx].keys/values = new_tensor
    - 必须使用 register_forward_pre_hook(with_kwargs=True)
    """

    def __init__(self, compression_ratio: float = 0.0, n_sink: int = 4):
        """
        Args:
            compression_ratio: 压缩率 (0.0-1.0)，例如 0.7 表示丢弃 70% 的中间 tokens
            n_sink: Attention Sinks，保留开头多少个 token 不被压缩
        """
        self.compression_ratio = compression_ratio
        self.n_sink = n_sink
        self.hooks = []
        self.compression_count = 0

        # 计算 max_capacity: 保留 sink + window 的总长度
        # 当压缩率是 0.7 时，保留 30% 的 tokens
        # 例如: n_sink=4, ratio=0.7 → 保留 30% → 假设原始 100 → 保留 30
        # 简化: 使用固定窗口 max_capacity
        # 这里为了兼容之前的 debug_press.py 逻辑，我们使用一个合理的默认值
        # 实际上压缩率应该动态计算，但为了简单我们先固定 max_capacity
        self.max_capacity = max(self.n_sink + 10, int(50 * (1 - compression_ratio)))

    def _make_hook(self, layer_idx):
        """为每个层创建专属的 hook"""

        def hook(module, args, kwargs):
            return self._pre_forward_hook(module, args, kwargs, layer_idx)

        return hook

    def _pre_forward_hook(self, module, args, kwargs, layer_idx):
        """Pre-forward hook: 拦截并压缩 DynamicCache 中的 KV"""
        # 查找 layer_past (DynamicCache 对象)
        cache = kwargs.get("layer_past")

        if cache is None:
            return args, kwargs

        # 检查是否是 DynamicCache
        if type(cache).__name__ != "DynamicCache":
            return args, kwargs

        # 使用 cache[layer_idx] 访问 KV tuple
        try:
            kv_tuple = cache[layer_idx]
        except:
            return args, kwargs

        if not isinstance(kv_tuple, tuple) or len(kv_tuple) != 2:
            return args, kwargs

        key, value = kv_tuple

        if key is None or value is None:
            return args, kwargs

        # seq_len 在维度 2: [batch, num_heads, seq_len, head_dim]
        seq_len = key.shape[2]

        # 判断是否需要压缩
        if seq_len <= self.max_capacity:
            return args, kwargs

        # 执行压缩
        self.compression_count += 1
        window_size = self.max_capacity - self.n_sink

        k_sink = key[:, :, : self.n_sink, :]
        v_sink = value[:, :, : self.n_sink, :]
        k_window = key[:, :, -window_size:, :]
        v_window = value[:, :, -window_size:, :]

        k_new = torch.cat([k_sink, k_window], dim=2)
        v_new = torch.cat([v_sink, v_window], dim=2)

        # 使用正确的方式修改 DynamicCache
        try:
            if hasattr(cache, "layers") and layer_idx < len(cache.layers):
                cache.layers[layer_idx].keys = k_new
                cache.layers[layer_idx].values = v_new
        except:
            pass

        return args, kwargs

    def register(self, model):
        """注册 Pre-Forward Hook 到模型的所有 Attention 层"""
        self.remove()

        # 适配 Pythia/GPT-NeoX 架构
        if hasattr(model, "gpt_neox"):
            layers = model.gpt_neox.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        else:
            raise ValueError("不支持的模型架构，找不到 layers")

        print(f"[StreamingLLM] 注册 Pre-Hook 到 {len(layers)} 个 Attention 层...")
        print(
            f"[StreamingLLM] 压缩率: {self.compression_ratio}, Sink: {self.n_sink}, MaxCapacity: {self.max_capacity}"
        )

        for i, layer in enumerate(layers):
            if hasattr(layer, "attention"):
                target = layer.attention
            elif hasattr(layer, "self_attn"):
                target = layer.self_attn
            else:
                continue

            # 使用 Pre-Forward Hook (with_kwargs=True)
            handle = target.register_forward_pre_hook(
                self._make_hook(i), with_kwargs=True
            )
            self.hooks.append(handle)

        print(f"[StreamingLLM] 成功注册 {len(self.hooks)} 个 Pre-Hook")

    def remove(self):
        """移除所有 Hook"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def __enter__(self):
        """支持 with press: 用法"""
        return self

    def __call__(self, model):
        """模拟 kvpress 的用法: with press(model):"""
        self.register(model)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
