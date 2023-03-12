import glob
from datetime import datetime

import torch

from args import args


def checkpoint_exist(root_path):
    checkpoints = glob.glob(root_path + "/checkpoints-*.pth")
    return len(checkpoints) > 0


def load_latest_checkpoint(root_path):
    checkpoints = glob.glob(root_path + "/checkpoints-*.pth")
    return load_checkpoint(checkpoints[-1])


def load_checkpoint(checkpoint_path):
    checkpoint_info = torch.load(checkpoint_path)
    return checkpoint_info


def save_checkpoint(net, optim, global_epoch, global_train_step):
    checkpoint_name = "checkpoints-{}.pth".format(datetime.now().strftime("%y%m%d-%H%M%S"))

    torch.save(
        {
            "net": net.state_dict(),
            "optim": optim.state_dict(),
            "global_epoch": global_epoch,
            "global_train_step": global_train_step,
        },
        args.checkpoint_path + "/" + checkpoint_name
    )

    print(f"save checkpoints {checkpoint_name}")
