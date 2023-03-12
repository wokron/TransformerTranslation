import re
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import vocab
from tqdm import tqdm

from args import args


def get_seq_and_vocab(file_path, tokenizer):
    token_freq = Counter()
    seqs = []

    with open(file_path, encoding="UTF-8") as f:
        with tqdm(f) as tqdm_file:
            tqdm_file.set_description(f"Load {file_path}")

            for idx, line in enumerate(tqdm_file):
                if args.debug and idx > 10000:
                    break

                tokens_in_line = tokenizer(line)
                tokens_in_line.insert(0, '<sos>')
                tokens_in_line.append('<eos>')

                seqs.append(tokens_in_line)

                token_freq += Counter(tokens_in_line)

    v = vocab(token_freq, 5, ['<pad>', '<unk>', '<sos>', '<eos>'])
    v.set_default_index(v['<unk>'])

    for idx in range(len(seqs)):
        seqs[idx] = [v[token] for token in seqs[idx]]

    return seqs, v


def zh_simple_tokenizer(line):
    return re.findall("[a-zA-Z]+|[^\s]", line)


class TranslateDataset(Dataset):
    def __init__(self, en_path, zh_path):
        en_seqs, self.en_vocab = get_seq_and_vocab(en_path, get_tokenizer("basic_english"))

        zh_seqs, self.zh_vocab = get_seq_and_vocab(zh_path, zh_simple_tokenizer)

        self.items = []
        for i in range(len(en_seqs)):
            en_seq = en_seqs[i]
            zh_seq = zh_seqs[i]

            src = en_seq
            tgt = zh_seq[:-1]
            pdt = zh_seq[1:]

            self.items.append((
                torch.as_tensor(src),
                torch.as_tensor(tgt),
                torch.as_tensor(pdt),
            ))

    def __getitem__(self, item):
        return self.items[item]

    def __len__(self):
        return len(self.items)
