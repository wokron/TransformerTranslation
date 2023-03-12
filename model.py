import math

import torch
from torch import nn

from args import args


def get_tgt_mask(size):
    mask = torch.tril(torch.ones(size, size) == 1).float()
    mask = mask.masked_fill(mask == 0, float("-inf"))
    mask = mask.masked_fill(mask == 1, float(0.0))
    mask = mask.to(args.device)

    return mask


def get_pad_mask(seq, pad_idx=0):
    return (seq == pad_idx).permute(1, 0)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(- torch.arange(0, embed_size, 2) * math.log(10000) / embed_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, embed_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, seq_embed):
        seq_embed_encoded = seq_embed + self.pos_embedding[:seq_embed.size(0), :]
        # (seq_len, batch_size, embed_size) + (seq_len, 1, embed_size)
        return self.dropout(seq_embed_encoded)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size

    def forward(self, seq):
        return self.embedding(seq) * math.sqrt(self.embed_size)


class TranslationModel(nn.Module):
    def __init__(self, vocab_size, embed_size, class_num, dropout):
        super(TranslationModel, self).__init__()
        self.embed = TokenEmbedding(vocab_size, embed_size)

        self.positional_encoding = PositionalEncoding(embed_size, dropout)

        self.transformer = nn.Transformer(
            embed_size,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=dropout,
        )

        self.multi = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Tanh(),
            nn.Linear(embed_size, class_num),
        )

    def forward(self, input):
        src, tgt = input  # (seq1_len, batch_size) (seq2_len, batch_size)
        src_embed = self.embed(src)  # (seq1_len, batch_size, embed_size)
        tgt_embed = self.embed(tgt)  # (seq2_len, batch_size, embed_size)

        src_embed_encoded = self.positional_encoding(src_embed)  # (seq1_len, batch_size, embed_size)
        tgt_embed_encoded = self.positional_encoding(tgt_embed)  # (seq2_len, batch_size, embed_size)

        tgt_mask = get_tgt_mask(tgt.shape[0])  # (seq2_len, seq2_len)
        src_pad_mask = get_pad_mask(src)  # (seq1_len, batch_size, embed_size)
        tgt_pad_mask = get_pad_mask(tgt)  # (seq2_len, batch_size, embed_size)

        out = self.transformer(
            src_embed_encoded,
            tgt_embed_encoded,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )  # (seq2_len, batch_size, embed_size)

        predict = self.multi(out)  # (seq2_len, batch_size, class_num)

        return predict
