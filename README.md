# NLPPJ

## Introduction

A Neural Machine Translation (NMT) project for Chinese-to-English translation, implementing RNN-Based and Transformer-based architectures.

## Requirements

The requirements.txt in every subproject folder contain the requirement. Use pip install -r requirements.txt to install it.

## Usage
The data cleaning and data preprocessing are included in the train.py.
### 1. Model Training
Activate the environment and change to the corresponding folder first.

#### Train Transformer Model

```
python train_transformer.py 

Key parameters:
- `--d_model`: Model dimension (128, 256, 512)
- `--n_heads`: Number of attention heads
- `--n_layers`: Number of encoder/decoder layers
- `--norm_type`: Normalization type (layernorm or rmsnorm)
you can edit it in the train.py files top.
```
#### Train RNN Model

```
python train_rnn.py 

Key parameters:
- `--rnn_type`: RNN type (gru or lstm)
- `--attention_type`: Attention type (dot, general, or additive)
- `--hidden_size`: Hidden layer size
- `--teacher_forcing`: Teacher forcing ratio
- `--decoding_type`: Greedy or beam-search decoding strategies
you can edit it in the train.py files top.
```

- ### 4. Model Inference

#### Transformer Inference

```
python inference.py 
```

#### RNN Inference

```
python test.py
```
