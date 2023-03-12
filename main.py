import os.path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from args import args
from checkpoint import checkpoint_exist, load_latest_checkpoint, save_checkpoint
from model import TranslationModel
from dataset import TranslateDataset

if os.path.exists(args.processed_dataset_path):
    dataset = torch.load(args.processed_dataset_path)
else:
    dataset = TranslateDataset(args.en_path, args.zh_path)
    torch.save(dataset, args.processed_dataset_path)

if args.debug:
    dev_size = 1000
    test_size = 1000
else:
    dev_size = 10000
    test_size = 10000
train_size = len(dataset) - dev_size - test_size

train_set, dev_set, test_set = random_split(
    dataset,
    [train_size, dev_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"train set: {len(train_set)}, dev_set: {len(dev_set)}, test_set: {len(test_set)}")


def collate_fn(data):
    src, tgt, pdt = zip(*data)
    src = pad_sequence(src)
    tgt = pad_sequence(tgt)
    pdt = pad_sequence(pdt)
    return [src, tgt], pdt


train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_set, args.batch_size, drop_last=True, collate_fn=collate_fn)
test_loader = DataLoader(test_set, args.batch_size, drop_last=True, collate_fn=collate_fn)

print("module:")

net = TranslationModel(
    len(dataset.en_vocab),
    args.embed_size,
    len(dataset.zh_vocab),
    args.dropout,
).to(args.device)

print(net)

writer = SummaryWriter(args.log_path)


def get_accuracy(predict, target):
    seq_size, batch_size, _ = predict.shape
    total_num = batch_size * seq_size
    accurate_num = (predict.topk(1)[1].squeeze() == target).sum().item()
    return accurate_num / total_num


def train(dataloader, net, criterion, optimizer):
    global global_train_step
    global global_epoch
    with tqdm(dataloader) as tqdm_loader:
        tqdm_loader.set_description(f"Epoch {global_epoch}")

        net.train()
        for input, target in tqdm_loader:
            global_train_step += 1

            for i in range(len(input)):
                input[i] = input[i].to(args.device)
            target = target.to(args.device)

            predict = net(input)

            loss = criterion(predict, target)

            accuracy = get_accuracy(predict, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tqdm_loader.set_postfix(loss=loss.item(), accuracy=accuracy)

            writer.add_scalar("train loss", loss.item(), global_train_step)
            writer.add_scalar("train accuracy", accuracy, global_train_step)


def show_example(predict, target):
    predict_seq = predict.topk(1)[1].squeeze()[:, 0]
    target_seq = target[:, 0]

    zh_itos = dataset.zh_vocab.get_itos()

    predict_str = ""
    for i in predict_seq.tolist():
        predict_str += zh_itos[i]

    target_str = ""
    for i in target_seq.tolist():
        target_str += zh_itos[i]

    print(f"predict: {predict_str}\ntarget: {target_str}")


def evaluate(tag, dataloader, net, criterion):
    net.eval()

    global global_epoch
    total_loss = 0
    accuracy = 0
    total_step = 0
    with torch.no_grad():
        for idx, (input, target) in enumerate(dataloader):
            for i in range(len(input)):
                input[i] = input[i].to(args.device)
            target = target.to(args.device)

            predict = net(input)

            if idx == 0:
                show_example(predict, target)

            loss = criterion(predict, target)

            total_loss += loss

            acc = get_accuracy(predict, target)

            accuracy += acc

            total_step += 1

    accuracy /= total_step
    total_loss /= total_step
    print(f"{tag}: loss={total_loss}, accuracy={accuracy}")

    writer.add_scalar(tag + " loss", total_loss, global_epoch)
    writer.add_scalar(tag + " accuracy", accuracy, global_epoch)


global_epoch = 0
global_train_step = 0

loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.zh_vocab['<pad>']).to(args.device)


def criterion(predict, target):
    return loss_fn(predict.permute(1, 2, 0), target.permute(1, 0))


optim = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

if checkpoint_exist(args.checkpoint_path):
    checkpoint = load_latest_checkpoint(args.checkpoint_path)
    net.load_state_dict(checkpoint["net"])
    optim.load_state_dict(checkpoint["optim"])
    global_epoch = checkpoint["global_epoch"]
    global_train_step = checkpoint["global_train_step"]


for param_group in optim.param_groups:
    param_group["lr"] = args.learning_rate


for i in range(args.max_epoch):
    global_epoch += 1

    train(train_loader, net, criterion, optim)

    save_checkpoint(net, optim, global_epoch, global_train_step)

    evaluate("dev", dev_loader, net, criterion)

    evaluate("test", test_loader, net, criterion)

writer.close()
