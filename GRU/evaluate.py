import torch
import random
from datapre import MAX_LENGTH, SOS_token, EOS_token
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 获取句子中每个单词的索引，返回的是索引序列
def indexesFromSentence(lang, sentence, tp):
    if tp =='en':
        return [lang.word2index[word] for word in sentence.split(' ')]
    else:
        return [lang.word2index[word] for word in list(sentence)]


# 根据索引序列建立张量，索引序列最后要添加终止符
def tensorFromSentence(lang, sentence, tp='ch'):
    # indexes = indexesFromSentence(lang, sentence, tp)
    # indexes.append(EOS_token)
    # if sentence[-1] != 2:
    #     sentence.append(EOS_token)
    t = copy.deepcopy(sentence)
    t.append(EOS_token)
    # return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
    return torch.tensor(t, dtype=torch.long, device=device).view(-1, 1)


# 输入张量是输入句子中单词的索引，输出张量是目标句子中单词的索引
def tensorsFromPair(tokenizer, pair):
    input_tensor = tensorFromSentence(tokenizer, pair[0], 'ch')
    target_tensor = tensorFromSentence(tokenizer, pair[1], 'en')
    return (input_tensor, target_tensor)


# 评估
def evaluate(tokenizer, encoder, decoder, sentence, max_length=MAX_LENGTH, idx2word={}):
    with torch.no_grad():
        input_tensor = tensorFromSentence(tokenizer, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                # decoded_words.append(tokenizer.id_to_token(EOS_token))
                # decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tokenizer.id_to_token(topi.item()))
                # decoded_words.append(topi.item())

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


# 随机从数据集选n个句子进行翻译测试
def evaluateRandomly(tokenizer, pairs, encoder, decoder, n=10):
    word2idx = tokenizer.get_vocab()
    idx2word = {idx: token for token, idx in word2idx.items()}
    targets = []
    predicts = []
    for i in range(n):
        pair = random.choice(pairs)
        decoded0 = []
        decoded1 = []
        # print(pair)
        # exit()
        for i in range(len(pair[0])):
            # decoded0.append(tokenizer.id_to_token(pair[0][i]))
            decoded0.append(pair[0][i])
        for i in range(len(pair[1])):
            decoded1.append(tokenizer.id_to_token(pair[1][i]))
            # decoded1.append(pair[1][i])
        # print('input:', decoded0)
        # print('target:', decoded1)
        output_words, attentions = evaluate(tokenizer, encoder, decoder, pair[0], idx2word=idx2word)
        targets.append(decoded1)
        predicts.append(output_words)
        # output_sentence = ' '.join(output_words)
        # print('predict', output_words)
        # print('')
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
    from nltk.tokenize import word_tokenize

    # 1. 将 ID 列表转为 token 列表（并过滤特殊符号）
    def ids_to_tokens(ids, idx2word, special_tokens=None):
        if special_tokens is None:
            special_tokens = {"<s>","<pad>","</s>","<unk>"}
        tokens = []
        if isinstance(ids, list):
            for i in ids:
                token = idx2word.get(i, "<unk>")
                if token != "<unk>":
                    tokens.append(token)
        else:
            for i in range(ids.shape[0]):
                token = idx2word.get(ids[i][0].item(), "<unk>")
                if token != "<unk>":
                    tokens.append(token)
        return tokens

    # 2. 转换所有 target 和 predict
    special_tokens = {"<s>","<pad>","</s>","<unk>"}
    list_of_references = []  # BLEU 要求每个 target 是一个 list of references
    hypotheses = []
    # print(targets[:3])
    # print(predicts[:3])
    # exit()
    for i in range(len(targets)):
        # ref_tokens = ids_to_tokens(targets[i], idx2word, special_tokens)
        # hyp_tokens = ids_to_tokens(predicts[i], idx2word, special_tokens)
        
        list_of_references.append([targets[i]])  # 注意：每个 reference 是一个列表的列表
        hypotheses.append(predicts[i])
    # print(list_of_references[:3])
    # print(hypotheses[:3])
    # exit()
    # 3. 计算 corpus-level BLEU（推荐）
    bleu = corpus_bleu(list_of_references, hypotheses)
    print(f"Corpus BLEU: {bleu:.4f}")

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()

