"""
下载 PG-19 数据集的单个 sample 到本地
用于 StreamingLLM 测试
"""
import os
from datasets import load_dataset
import json

def download_pg19_sample():
    """下载 PG-19 的第一个测试样本"""
    print("=" * 60)
    print("开始下载 PG-19 单个 sample")
    print("=" * 60)
    
    # 设置保存路径
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, "hf_cache", "datasets", "pg19_sample")
    os.makedirs(save_dir, exist_ok=True)
    
    # 配置 HuggingFace 镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HOME"] = os.path.join(current_dir, "hf_cache")
    
    try:
        print("\n正在连接 HuggingFace Hub...")
        print("使用镜像: https://hf-mirror.com")
        
        # 使用 streaming 模式只下载第一个 sample
        print("\n正在下载 PG-19 test split 的第一个 sample...")
        ds = load_dataset(
            "pg19", 
            split="test",
            streaming=True,
            trust_remote_code=True
        )
        
        # 获取第一个样本
        sample = next(iter(ds))
        
        # 保存为 JSON 和纯文本两种格式
        json_path = os.path.join(save_dir, "pg19_sample.json")
        txt_path = os.path.join(save_dir, "pg19_sample.txt")
        
        # 保存 JSON（包含所有字段）
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
        
        # 保存纯文本（只保存 text 字段）
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(sample['text'])
        
        # 统计信息
        text_length = len(sample['text'])
        word_count = len(sample['text'].split())
        
        print("\n✅ 下载成功！")
        print(f"  保存路径: {save_dir}")
        print(f"  JSON 文件: pg19_sample.json")
        print(f"  文本文件: pg19_sample.txt")
        print(f"\n📊 样本信息:")
        print(f"  字符数: {text_length:,}")
        print(f"  单词数: {word_count:,}")
        print(f"  文件大小: {os.path.getsize(txt_path) / 1024:.1f} KB")
        
        if 'short_book_title' in sample:
            print(f"  书名: {sample['short_book_title']}")
        
        print("\n" + "=" * 60)
        print("下载完成！现在可以运行 python main.py")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的原因:")
        print("  1. 网络连接问题")
        print("  2. HuggingFace 访问受限")
        print("  3. 镜像站点暂时不可用")
        print("\n建议:")
        print("  - 检查网络连接")
        print("  - 稍后重试")
        print("  - 或使用 VPN 访问")
        return False

if __name__ == "__main__":
    success = download_pg19_sample()
    
    if success:
        print("\n🎉 可以开始运行测试了:")
        print("   python main.py")
    else:
        print("\n⚠️  请解决网络问题后重试")
