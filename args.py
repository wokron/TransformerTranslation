from argparse import Namespace

import torch

args = Namespace(
    batch_size=128,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    debug=True,
    max_epoch=40,
    learning_rate=0.0001,

    embed_size=128,
    dropout=0.1,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=512,


    zh_path="./data/chinese.zh",
    en_path="./data/english.en",
    processed_dataset_path="./data/dataset.pth",
    log_path="./logs",
    checkpoint_path="./checkpoints",
)
