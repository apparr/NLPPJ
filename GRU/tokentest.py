# 这也不是恳求信任任何拥有权威“气场”的人。
# 马德里—美国正在为最令人兴奋（也最令人疲惫）的政治事件做准备：总统席位的开放竞争。
# MADRID – The United States is gearing up for that most intoxicating (and exhausting) of political events: an open-seat race for the presidency.

import os
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

def train_bpe_tokenizer(files, vocab_size=10000, save_path="./bpe_vocab"):
    # 1. Initialize
    tokenizer = ByteLevelBPETokenizer()
    
    print(f"Training BPE Tokenizer on {files}...")
    print(f"Target Vocab Size: {vocab_size}")

    # 2. Train
    # min_frequency=2: 至少出现2次的字符组合才会被合并
    tokenizer.train(files=files, vocab_size=vocab_size, min_frequency=2, special_tokens=[
        "<s>",      # SOS
        "<pad>",    # PAD
        "</s>",     # EOS
        "<unk>",    # UNK
    ])
    
    # 3. Post-Process (Add special tokens automatically)
    # 这让 tokenizer.encode("hello").ids 自动加上 <s> 和 </s>
    # 格式: <s> body </s>
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    
    # 4. Save
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    tokenizer.save_model(save_path)
    print(f"Tokenizer saved to {save_path}")

if __name__ == "__main__":
    # 我们用训练集的中英文数据一起来训练共享词表
    # 为了方便，我们先临时合并一下数据（或者直接传入 jsonl 读取逻辑比较麻烦，直接把文本导出来训练最快）
    
    import json
    
    # 提取纯文本供训练
    raw_text_file = "./temp/temp_train_corpus.txt"
    with open('/data/pj/AP0004_MidtermFinal_translation_dataset_zh_en/AP0004_Midterm&Final_translation_dataset_zh_en/train_100k.jsonl', 'r', encoding='utf-8') as f, \
         open(raw_text_file, 'w', encoding='utf-8') as out:
        for line in f:
            item = json.loads(line)
            out.write(item['zh'] + "\n")
            out.write(item['en'] + "\n")
            
    # 100k 数据量不大，Vocab 给 8000-15000 足够了。太大会导致 embedding 层太大。
    # 这里我们选 10000
    train_bpe_tokenizer([raw_text_file], vocab_size=8000)
    
    # 清理临时文件
    os.remove(raw_text_file)

    # from tokenizers import ByteLevelBPETokenizer
    # tokenizer = ByteLevelBPETokenizer(
    #     "./bpe_vocab/vocab.json",
    #     "./bpe_vocab/merges.txt"
    # )
    # tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
    # encoded = tokenizer.encode("PARIS – As the economic crisis deepens and widens, the world has been searching for historical analogies to help us understand what has been happening.")
    # print(encoded.tokens)
    # print(encoded.ids)  # [0, ..., 2]  # <s> ... </s>