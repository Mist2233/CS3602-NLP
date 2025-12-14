import os
import time
import torch
import math
from tqdm import tqdm

# 1. 依然要设置镜像，否则下载数据集会报错
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

current_dir = os.getcwd()
os.environ["HF_HOME"] = os.path.join(current_dir, "hf_cache")

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ================= 配置区域 =================
MODEL_PATH = "./models/pythia-70m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"检测到的设备: {DEVICE}")

MAX_LENGTH = 2048  # Pythia 的最大上下文窗口
# ===========================================

print(f"正在加载模型: {MODEL_PATH} 到 {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"  # 使用半精度加速
)
model.eval()  # 开启评估模式


# -------- 定义计算 PPL 的函数 --------
def calculate_ppl(text, stride=512):
    """
    计算长文本的 PPL (Perplexity)。
    由于文本可能超过模型最大长度，需要用滑动窗口(stride)切分。
    """
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    print(f"文本总长度: {seq_len} tokens, 开始计算 PPL...")

    # 这是一个标准的滑动窗口计算 PPL 的循环
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + MAX_LENGTH, seq_len)
        trg_len = end_loc - prev_end_loc  # 这里的逻辑是为了处理重叠部分

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()

        # 我们不计算重叠部分的 loss，只计算新部分的
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # outputs.loss 是平均 log-likelihood
            # 我们需要乘回去得到总的 loss
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # 最终计算 PPL = exp(总loss / 总长度)
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()


# -------- 定义测试生成速度的函数 --------
def test_speed(input_text, generate_len=50):
    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
    print("开始速度测试 (生成 50 个 token)...")

    start_time = time.time()
    with torch.no_grad():
        model.generate(
            **inputs, max_new_tokens=generate_len, pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()

    duration = end_time - start_time
    speed = generate_len / duration
    print(f"耗时: {duration:.4f}s, 速度: {speed:.2f} tokens/s")
    return speed


# ================= 任务 1: WikiText 测试 =================
print("\n" + "=" * 20 + " 测试 WikiText-2 " + "=" * 20)
# 加载 wikitext-2 测试集
wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# 为了快速演示，我们把所有测试数据拼成一个长字符串（标准做法之一）
wiki_text = "\n\n".join(wiki_data["text"])

print(f"WikiText 字符数: {len(wiki_text)}")
ppl = calculate_ppl(wiki_text)
print(f"WikiText-2 PPL: {ppl:.2f}")


# ================= 任务 2: PG-19 测试 (单一样本) =================
print("\n" + "=" * 20 + " 测试 PG-19 (单本) " + "=" * 20)
# streaming=True 表示不下载整个数据集（几百G），而是像看视频一样边下边看
pg19_stream = load_dataset(
    "pg19",
    split="test",
    streaming=True,
    trust_remote_code = True,
)

# 取出第一本书作为 sample
book_sample = next(iter(pg19_stream))
book_text = book_sample["text"]

# PG-19 的书可能极长，为了测试不跑太久，我们只取前 10000 个字符做 PPL 测试
# 如果你想测整本书，把 [:10000] 去掉即可
short_book_text = book_text[:10000]

print(f"书名: {book_sample.get('short_book_title', 'Unknown')}")
print(f"截取长度: {len(short_book_text)} 字符 (原书超长)")

# 1. 测 PPL
pg_ppl = calculate_ppl(short_book_text)
print(f"PG-19 Sample PPL: {pg_ppl:.2f}")

# 2. 测速度 (用开头的一句话作为 prompt)
prompt = book_text[:100]
test_speed(prompt)

print("\nBaseline 测试完成！")
