import json
import re
import jieba
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize

# 首次运行需下载 punkt
# nltk.download('punkt')

# text = "1929 or 1989?"
# tokens = word_tokenize(text)
# import spacy

# # 加载模型（只需一次）
# nlp = spacy.load("en_core_web_sm")

# def lemmatize_sentence(text):
#     doc = nlp(text)
#     return [token.lemma_.lower() for token in doc if not token.is_space]

# ================================
# 1. Data Cleaning
# ================================
def clean_text(text, tp):
    """清理文本：移除非法字符、多余空格、换行等"""
    # 移除非法字符（只保留字母、数字、中文字符、标点和空格）
    text = re.sub(r'[^\w\u4e00-\u9fff\s\.\,\!\?\;\:\'\"-_()–，。-（）——：？；’‘”“！]', '', text) #修改版
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    if tp == 'en':
        text = text.lower().strip()
        text = re.sub(r'([.,!?;:()\'\"])', r' \1 ', text)  # 标点前后加空格
        text = re.sub(r'\s+', ' ', text)              # 合并多个空格
    return text


def filter_rare_words_and_long_sentences(
    src_texts, tgt_texts,
    max_length=49,
    min_freq=2,
    vocab_size_limit=30000):
    """过滤稀有词和过长句子"""
    # 统计词频
    # src_words = []
    # tgt_words = []

    # for src in src_texts:
    #     src_words.extend(list(src))
    # for tgt in tgt_texts:
    #     tgt_words.extend(list(tgt.split()))
    # # print(tgt_words[:100])
    # # print(src_words[:100])
    # # exit()

    # src_counter = Counter(src_words)
    # tgt_counter = Counter(tgt_words)
    # # print(tgt_counter)
    # # exit()
    # # 过滤低频词
    filtered_src = []
    filtered_tgt = []

    t = 0
    for i in range(len(src_texts)):
        if len(src_texts[i]) >= max_length or len(tgt_texts[i]) >= max_length:
            continue  # 截断过长句子
        # 过滤稀有词（替换为 <UNK>）
        # src_filtered = [word if src_counter[word] >= min_freq else '<UNK>' for word in list(src)]
        # tgt_filtered = [word if tgt_counter[word] >= min_freq else '<UNK>' for word in list(tgt.split())]
        filtered_src.append(src_texts[i])
        filtered_tgt.append(tgt_texts[i])
        
    # for tgt in tgt_texts:
    #     if len(tgt) >= max_length:
    #         continue  # 截断过长句子
    #     # 过滤稀有词（替换为 <UNK>）
    #     # src_filtered = [word if src_counter[word] >= min_freq else '<UNK>' for word in list(src)]
    #     # tgt_filtered = [word if tgt_counter[word] >= min_freq else '<UNK>' for word in list(tgt.split())]
    #     filtered_tgt.append(tgt)

    return  filtered_src, filtered_tgt


# ================================
# 2. Tokenization
# ================================
class Tokenizer:
    def __init__(self, language="en", use_jieba=True, vocab_size=None):
        self.language = language
        self.use_jieba = use_jieba
        self.vocab = None
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = vocab_size

    def tokenize(self, text):
        """对单个句子进行分词"""
        if self.language == "zh":
            if self.use_jieba:
                # print(list(text))
                # exit()
                return list(jieba.cut(text))
            else:
                return list(text)  # 字符级
        elif self.language == "en":
            # 使用空格或 BPE（此处简化为空格分词）
            return list(text.split())

    def build_vocab(self, texts, min_freq=2):
        """从文本列表构建词表"""
        word_counter = Counter()
        if self.language == "en":
            for text in texts:
                tokens = self.tokenize(text)
                tokens = [token for token in tokens if token != ' ']
                # print(tokens)
                # t=t+1
                # if t==2:
                #     exit()
                word_counter.update(tokens)
        else:
            for text in texts:
                word_counter.update(text)

        # 过滤低频词
        vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        for word, freq in word_counter.most_common():
            if freq >= min_freq and word not in vocab:
                vocab.append(word)

        self.vocab = vocab
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        self.vocab_size = len(vocab)
        print(f"Built {self.language} vocabulary with {len(vocab)} words")
        # print(vocab[:30])

    def encode(self, text):
        """将文本编码为 token ID 列表"""
        if self.language == "en":
            tokens = self.tokenize(text)
            return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
        else:
            return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in text]

    def decode(self, ids):
        """将 token ID 解码为文本"""
        return " ".join([self.idx2word.get(idx, "<UNK>") for idx in ids])


# ================================
# 3. Vocabulary Construction
# ================================
def build_vocabs(src_texts, tgt_texts, min_freq=2, vocab_size_limit=30000):
    """构建源语言和目标语言的词表"""
    src_tokenizer = Tokenizer(language="zh", use_jieba=False)
    tgt_tokenizer = Tokenizer(language="en", use_jieba=False)

    src_tokenizer.build_vocab(src_texts, min_freq=min_freq)
    tgt_tokenizer.build_vocab(tgt_texts, min_freq=min_freq)

    # 控制词汇表大小
    if src_tokenizer.vocab_size > vocab_size_limit:
        src_tokenizer.vocab = src_tokenizer.vocab[:vocab_size_limit]
        src_tokenizer.word2idx = {word: idx for idx, word in enumerate(src_tokenizer.vocab)}
        src_tokenizer.idx2word = {idx: word for idx, word in enumerate(src_tokenizer.vocab)}

    return src_tokenizer, tgt_tokenizer


# ================================
# 5. Implementation Flexibility: 支持多种策略
# ================================
def preprocess_data(file_path, max_length=45, min_freq=2, vocab_size_limit=30000):
    """
    主函数：完整数据预处理流程
    """
    # Step 1: Load and Clean Data
    print("Loading data...")
    data = load_your_data(file_path)  # 替换为你自己的加载函数

    src_texts = [item["zh"] for item in data]
    tgt_texts = [item["en"] for item in data]
    # print(len(src_texts)) 100000

    # Step 2: Clean Text
    print("Cleaning texts...")
    src_cleaned = [clean_text(text, 'zh') for text in src_texts]
    tgt_cleaned = [clean_text(text, 'en') for text in tgt_texts]

    # Step 3: Filter Rare Words & Long Sentences
    # print("Filtering rare words and long sentences...")
    # src_filtered, tgt_filtered = filter_rare_words_and_long_sentences(
    #     src_cleaned, tgt_cleaned,
    #     max_length=max_length,
    #     min_freq=min_freq,
    #     vocab_size_limit=vocab_size_limit
    # )
    # print(list(tgt_filtered[1].split()))
    # exit()
    # print(src_filtered[:10])
    # exit()
    # Step 4: Build Vocabularies
    print("Building vocabularies...")
    # src_tokenizer, tgt_tokenizer = build_vocabs(src_filtered, tgt_filtered, min_freq=min_freq)
    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer(
        "./bpe_vocab/vocab.json",
        "./bpe_vocab/merges.txt"
    )
    tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
    # encoded = tokenizer.encode("")
    # print(encoded.tokens)
    # print(src_filtered[1])
    # exit()
    # Step 5: Encode Data
    print("Encoding data...")
    src_encoded = [tokenizer.encode(text).ids for text in src_cleaned]
    tgt_encoded = [tokenizer.encode(text).ids for text in tgt_cleaned]

    print("Filtering rare words and long sentences...")
    src_filtered, tgt_filtered = filter_rare_words_and_long_sentences(
        src_encoded, tgt_encoded,
        max_length=max_length,
        min_freq=min_freq,
        vocab_size_limit=vocab_size_limit
    )
    # print(src_encoded[:10])
    # exit()
    # Step 6: Pad sequences to same length
    # max_src_len = max(len(seq) for seq in src_encoded)
    # max_tgt_len = max(len(seq) for seq in tgt_encoded)

    # src_padded = [seq + [src_tokenizer.word2idx["<PAD>"]] * (max_src_len - len(seq)) for seq in src_encoded]
    # tgt_padded = [seq + [tgt_tokenizer.word2idx["<PAD>"]] * (max_tgt_len - len(seq)) for seq in tgt_encoded]

    # Convert to tensors
    # src_tensor = torch.tensor(src_encoded)
    # tgt_tensor = torch.tensor(tgt_encoded)

    return {
        "src_tensor": src_filtered,
        "tgt_tensor": tgt_filtered,
        # "src_tokenizer": src_tokenizer,
        # "tgt_tokenizer": tgt_tokenizer,
        # "src_vocab_size": src_tokenizer.vocab_size,
        # "tgt_vocab_size": tgt_tokenizer.vocab_size,
        "tokenizer": tokenizer
    }


# ================================
# Helper: Load Your Data
# ================================
def load_your_data(file_path="data.jsonl"):
    """从 JSONL 文件加载数据"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "zh" in item and "en" in item:
                    data.append(item)
            except Exception as e:
                print(f"Error parsing line: {line} | {e}")
    return data


# ================================
# Example Usage
# ================================
if __name__ == "__main__":
    # 预处理数据
    result = preprocess_data("/data/pj/AP0004_MidtermFinal_translation_dataset_zh_en/AP0004_Midterm&Final_translation_dataset_zh_en/train_100k.jsonl", max_length=50, min_freq=2)

    print(f"Source vocab size: {result['src_vocab_size']}")
    print(f"Target vocab size: {result['tgt_vocab_size']}")
    print(f"Number of samples: {len(result['src_tensor'])}")
    print(f"Number of samples: {len(result['tgt_tensor'])}")

    # 查看第一个样本
    print("Sample source:", result['src_tokenizer'].decode(result['src_tensor'][0]))
    print("Sample target:", result['tgt_tokenizer'].decode(result['tgt_tensor'][0]))