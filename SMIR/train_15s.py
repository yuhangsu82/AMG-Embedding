import os
import glob
from pathlib import Path
import random
import numpy as np
from prettytable import PrettyTable
from config import Config
import torch
from torch import nn
from torch.utils import data as Data
from tqdm import tqdm
from models.mt import MT
from models.mlstm import MLSTM
from models.mcnn import MCNN
from models.mt_test import MT1
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm
import logging
from losses import ArcFaceLoss
import argparse
import faiss
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle
from evaluate import eval
from utils import generate_chunk


def init_model(model_type):
    if model_type == 'mt':
        model = MT(
            input_dim=128,
            output_dim=128,
            dim=256,
            depth=10,
            heads=8,
            dim_head=16,
            mlp_dim=256,
            dropout=0.1,
        ).cuda()
        # model = MT(
        #     input_dim=128,
        #     output_dim=opt.feature_dim,
        #     dim=256,
        #     depth=16,
        #     heads=64,
        #     dropout=0.1,
        # ).cuda()
    elif model_type == 'mt_test':
        model = MT1(
            input_dim=128,
            output_dim=opt.feature_dim,
            dim=256,
            depth=10,
            heads=8,
            dropout=0.1,
        ).cuda()
    elif model_type == 'mlstm':
        model = MLSTM(128, opt.feature_dim, 256, 8, True, 0.1).cuda()
    elif model_type == 'mcnn':
        model = MCNN(opt.feature_dim).cuda()
    else:
        raise Exception("No such model!")

    return model


def load_file(input_file_path):
    files = list()
    for filename in os.listdir(input_file_path):
        if int(filename.split("-")[0]) <= opt.class_nums and int(filename.split('-')[1][:-4]) < 10:
        # if int(filename.split("-")[0]) <= opt.class_nums:
            files.append(filename)
    return files


class SpectrogramFingerprintData(Data.Dataset):
    def __init__(self, files, data_path):
        self.files = files
        self.data_path = data_path

    def __getitem__(self, index):
        file = self.files[index]
        path = os.path.join(self.data_path, file)
        data = np.load(path)
        label = int(file.split("-")[0]) - 1
        length = min(len(data), 32)
        start_index, chunk_len = generate_chunk(length)
        if int(file.split("-")[1].split(".")[0]) == 0:
            start_index = 0
            chunk_len = len(data)
        sample_weight = 1
        margin = 0.2 * chunk_len / len(data) + 0.2
        # margin = 0.2
        sequence = np.zeros((32, 128), np.float32)
        padding_mask = np.zeros((32), dtype=np.bool_)   
        padding_mask[chunk_len:] = 1
        sequence[:min(chunk_len, 32)] = data[start_index: min(start_index + chunk_len, start_index + 32)]

        return np.float32(sequence), np.int64(label), np.float32(margin), np.bool_(padding_mask)

    def __len__(self):
        return len(self.files)


class RetrievalData(Data.Dataset):
    def __init__(self, paths, root_dir, max_len=32, start_id=0):
        self.paths = paths
        self.root_dir = root_dir
        self.max_len = max_len
        self.start_id = start_id

    def __getitem__(self, index):       
        file_path = self.paths[index]
        data = np.load(self.root_dir + "/" + file_path)

        sequence = np.zeros((32, 128), dtype=np.float32)
        length = min(len(data), 32)

        if self.max_len == 32:
            chunk_len = length
        else:
            chunk_len = self.max_len
        
        for i in range(min(length, self.max_len)):
            sequence[i] = data[i + self.start_id]

        padding_mask = np.zeros((32), dtype=np.bool_)
        padding_mask[chunk_len:] = 1

        return np.float32(sequence), np.bool_(padding_mask)

    def __len__(self):
        return len(self.paths)   


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
        datefmt="%a %b %d %H:%M:%S %Y",
    )
    work_dir = os.path.join(f"./log/{opt.model}", time.strftime("%Y-%m-%d-%H.%M", time.localtime()))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fHandler = logging.FileHandler(work_dir + "/log.txt", mode="w")
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)
    return logger


def train():
    model.train()
    train_nb = len(train_data)
    total_num_train = len(train_data.dataset)
    loss_accumulation = loss_avg = correct_num = 0
    pbar_train = tqdm(train_data, total=train_nb)
    for step, (tensor, label, margin, padding_mask) in enumerate(pbar_train):
        tensor = tensor.cuda()
        label = label.cuda()
        margin = margin.cuda()  
        padding_mask = padding_mask.cuda()
        
        local_embedding = model(tensor, padding_mask)
        # local_embedding = model(tensor)
        logit, loss = arcface(local_embedding, label, margin)   
        if loss_accumulation is None and loss_avg is None:
            loss_accumulation = loss.item()
            loss_avg = loss.item()
        else:
            loss_accumulation += loss.item()
            loss_avg = loss_accumulation / (step + 1)
        pred = torch.argmax(logit, dim=1)
        correct_num += torch.sum(pred == label).item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        current_lr = scheduler.get_last_lr()[0]
        s = "train ===> epoch:{} ---- step:{} ----lr:{}---- loss:{:.4f} ---- loss_avg:{:.4f}".format(
            epoch, step, current_lr, loss, loss_avg
        )
        pbar_train.set_description(s)
        logger.info(s)

    accuracy_train = correct_num / total_num_train
    return loss_avg, accuracy_train


def init_arcface_center_128():
    center = np.zeros((opt.class_nums, 128), dtype=np.float32)
    for i in range(1, opt.class_nums + 1):
        data = np.load(opt.dataset_path + f"/{i}-00.npy")
        center[i-1] = np.mean(data, axis=0)
    t_c = torch.from_numpy(center).float().cuda()
    with torch.no_grad():
        arcface.weight.data = t_c
    print("Init arcface center successfully!")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='./datasets/mt/train/data')
    parser.add_argument("--model", type=str, default='mt_test')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--last_lr", type=float, default=0.00001)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--class_nums", type=int, default=20000)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    

    return parser.parse_args()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt = parse_opt()
    model = init_model(opt.model)
    logger = get_logger()
    log_dir = f"./runs/checkpoint/{opt.model}_arcface/exp/{opt.model}.pth"

    train_files = load_file(opt.dataset_path)

    train_data = SpectrogramFingerprintData(files=train_files, data_path=opt.dataset_path)
    train_data = Data.DataLoader(train_data, shuffle=True, batch_size=opt.batch_size)
    arcface = ArcFaceLoss(in_features=opt.feature_dim, out_features=opt.class_nums, s=64.0).cuda()
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': arcface.parameters()}], lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch * len(train_data), eta_min=opt.last_lr)

    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint["model"])
        arcface.load_state_dict(checkpoint["arcface"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        print("Loaded epoch {} successfully!".format(start_epoch))
    else:
        start_epoch = 0
        print("Train the model from scratch without saving previous model!")
        df_train = pd.DataFrame(
            columns=[
                "time",
                "epoch",
                "train/loss",
                "train/accuracy",
            ]
        )
        df_train.to_csv(f"./runs/checkpoint/{opt.model}_arcface/exp/result.csv", index=False)
        start_epoch = 0


    for epoch in range(start_epoch + 1, opt.epoch + 1):
        if epoch == 1:
            init_arcface_center_128()

        train_loss, train_acc = train()
        time_now = time.strftime("%Y-%m-%d-%H.%M", time.localtime())
        data_list = [time_now, epoch, train_loss,train_acc]
        data = pd.DataFrame([data_list])
        data.to_csv(f"./runs/checkpoint/{opt.model}_arcface/exp/result.csv", mode="a", header=False, index=False)

        state = {
            "model": model.state_dict(),
            "arcface": arcface.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch
        }
        if epoch % 5 == 0:
            print("epoch: ", epoch)
            table_string = eval(
                emb_dir='./runs/retrieval/test_15',
                emb_dummy_dir='./database/fma_part_15s',
                model_type=opt.model,
                checkpoint_path=log_dir,
                max_len=32,
                device='cuda:0',
                batch_size=1500,
                feature_dim=128,
                k_prob=10,
            )

            with open(f"./runs/checkpoint/{opt.model}_arcface/exp/table.txt", "a") as file:
                file.write(f"epoch: {epoch}\n")
                file.write(table_string + "\n")
            
            

        torch.save(state, log_dir)
        torch.save(state, f"./runs/checkpoint/{opt.model}_arcface/exp/{opt.model}_{epoch}.pth".format(epoch))
        
