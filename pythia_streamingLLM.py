from streaming_llm_standalone import load_model_and_tokenizer, StreamingLLM
import time
import torch


def benchmark_speed_streaming_kv(
    model,
    tokenizer,
    prompt: str,
    n_sink: int = 4,
    window_size: int = 256,
    num_tokens: int = 50,
    batch_size: int = 1,
):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(next(model.parameters()).device)
    input_ids = input_ids.repeat(batch_size, 1)

    stream = StreamingLLM(n_sink=n_sink, window_size=window_size)

    start = time.time()
    stream.generate(model, input_ids, max_new_tokens=num_tokens)
    end = time.time()

    duration = end - start
    total_tokens = num_tokens * batch_size
    tok_per_sec = total_tokens / duration
    return tok_per_sec

def evaluate_ppl_streaming_kv(
    model,
    tokenizer,
    text: str,
    n_sink: int = 4,
    window_size: int = 256,
    max_tokens: int = 2000,
):
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[:, :max_tokens].to(next(model.parameters()).device)
    seq_len = input_ids.size(1)
    if seq_len < 2:
        return float("inf")

    stream = StreamingLLM(n_sink=n_sink, window_size=window_size)
    cache = None
    losses = []
    ce = torch.nn.CrossEntropyLoss(reduction="none")

    for pos in range(seq_len - 1):
        cur = input_ids[:, pos:pos + 1]
        with torch.no_grad():
            if cache is None:
                outputs = model(cur, use_cache=True)
            else:
                outputs = model(cur, use_cache=True, past_key_values=cache)
            cache = outputs.past_key_values
            stream.compress_cache(cache)
            logits = outputs.logits[:, -1, :]
            target = input_ids[:, pos + 1]
            loss = ce(logits, target).mean()
            losses.append(loss)

    ppl = torch.exp(torch.stack(losses).mean())
    return ppl.item()

model_id = "EleutherAI/pythia-2.8b"
model, tokenizer = load_model_and_tokenizer(model_id)

prompt = "在一座海边小城里，工程师正在测试一种新的 KV 缓存压缩算法。"

# 生成效果看看
encoded = tokenizer(prompt, return_tensors="pt")
input_ids = encoded.input_ids
stream = StreamingLLM(n_sink=4, window_size=256)
out_ids = stream.generate(model, input_ids, max_new_tokens=50)
print(repr(tokenizer.decode(out_ids[0], skip_special_tokens=False)))

with open("custom_complex_dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

ppl_stream = evaluate_ppl_streaming_kv(
    model,
    tokenizer,
    text,
    n_sink=4,
    window_size=256,
    max_tokens=2000,
)
print("StreamingLLM PPL on custom_complex_dataset:", ppl_stream)

# 测速度
speed = benchmark_speed_streaming_kv(
    model,
    tokenizer,
    prompt,
    n_sink=4,
    window_size=256,
    num_tokens=50,
    batch_size=1,
)
print("StreamingLLM KV-level speed (tok/s):", speed)