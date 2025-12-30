import torch
from datapre import prepareData
from seq2seq_model import EncoderRNN, AttnDecoderRNN
from evaluate import evaluateRandomly as evaluate
from datapre2 import preprocess_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




if __name__ == '__main__':
    data = preprocess_data("/data/pj/AP0004_MidtermFinal_translation_dataset_zh_en/AP0004_Midterm&Final_translation_dataset_zh_en/train_100k.jsonl")
    src_data = data["src_tensor"]      # list of list of int
    tgt_data = data["tgt_tensor"]
    # src_vocab_size = data["src_vocab_size"]
    # tgt_vocab_size = data["tgt_vocab_size"]
    # input_lang = data["src_tokenizer"]
    # output_lang = data["tgt_tokenizer"]
    tokenizer = data["tokenizer"]
    pairs = []
    for i in range(len(src_data)):
        pairs.append([src_data[i], tgt_data[i]])
    hidden_size = 256  # 隐藏层维度设置为256
    encoder1 = EncoderRNN(tokenizer.get_vocab_size(), hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, tokenizer.get_vocab_size(), dropout_p=0.1).to(device)
    # 恢复网络
    encoder1.load_state_dict(torch.load('encoder100000.pkl'))
    attn_decoder1.load_state_dict(torch.load('decoder100000.pkl'))

    evaluate(tokenizer, pairs, encoder1, attn_decoder1)

    # # 输入一些句子测试下
    # evaluateAndShowAttention("elle a cinq ans de moins que moi .")
    # evaluateAndShowAttention("elle est trop petit .")
    # evaluateAndShowAttention("je ne crains pas de mourir .")
    # evaluateAndShowAttention("c est un jeune directeur plein de talent .")