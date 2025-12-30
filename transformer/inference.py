import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from terminaltables import AsciiTable

from AttModel import AttModel
from bleu import bleu
from data_load import (
    get_batch_indices,
    load_cn_vocab,
    load_en_vocab,
    load_test_data,
    load_train_data,
)
from hyperparams import Hyperparams as hp
from util import get_logger
from datapre import preprocess_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def evaluate(model_dir):
    # Load data
    data = preprocess_data("/data/pj/AP0004_MidtermFinal_translation_dataset_zh_en/AP0004_Midterm&Final_translation_dataset_zh_en/test.jsonl")
    src_data = data["src_tensor"]      # list of list of int
    tgt_data = data["tgt_tensor"]
    tokenizer = data["tokenizer"]
    vocab = tokenizer.get_vocab()
    id2token = {id_: token for token, id_ in vocab.items()}
    global_batches = 0

    cn2idx = en2idx = tokenizer.get_vocab()
    idx2cn = idx2en = id2token
    enc_voc = len(cn2idx)
    dec_voc = len(en2idx)
    model = AttModel(hp, enc_voc, dec_voc)
    model.load_state_dict(torch.load(model_dir))

    X = load_test_data(src_data, tgt_data)
    #Sources, Targets 
    Sources = [tokenizer.decode(ids, skip_special_tokens=True) for ids in src_data]
    Targets = [tokenizer.decode(ids, skip_special_tokens=True) for ids in tgt_data]
    vocab = tokenizer.get_vocab()
    id2token = {id_: token for token, id_ in vocab.items()}
    global_batches = 0

    cn2idx = en2idx = tokenizer.get_vocab()
    idx2cn = idx2en = id2token

    model.to(device)
    # Inference
    if not os.path.exists("results"):
        os.mkdir("results")
    list_of_refs = []
    hypotheses = []
    assert hp.batch_size_valid <= len(
        X
    ), "test batch size is large than total data length. Check your data or change batch size."

    for i in range(len(X) // hp.batch_size_valid):
        # Get mini-batches
        x = X[i * hp.batch_size_valid : (i + 1) * hp.batch_size_valid]
        sources = Sources[i * hp.batch_size_valid : (i + 1) * hp.batch_size_valid]
        targets = Targets[i * hp.batch_size_valid : (i + 1) * hp.batch_size_valid]

        # Autoregressive inference
        x_ = torch.LongTensor(x).to(device)
        preds_t = torch.LongTensor(
            np.zeros((hp.batch_size_valid, hp.maxlen), np.int32)
        ).to(device)
        preds = preds_t
        _, _preds, _ = model(x_, preds)
        preds = _preds.data.cpu().numpy()

        # prepare data for BLEU score
        for source, target, pred in zip(sources, targets, preds):
            got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
            ref = target.split()
            hypothesis = got.split()
            if len(ref) > 3 and len(hypothesis) > 3:
                list_of_refs.append([ref])
                hypotheses.append(hypothesis)

if __name__ == '__main__':
    model_dir = './models/model.pth'
    evaluate(model_dir)