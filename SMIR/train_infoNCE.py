import os
import glob
from pathlib import Path
import random
import ast
import numpy as np
from config import Config
import torch
from torch import nn
from torch.utils import data as Data
from tqdm import tqdm
from models.mt import MT
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
import logging
from losses.InfoNCE import InfoNCELoss
import faiss
# from torch.optim.lr_scheduler import CosineAnnealingLR


def load_file_org(input_file_path):
    files = list()
    for filename in os.listdir(input_file_path):
        if filename.split("-")[1] == "00.npy":
            files.append(filename)
    return files


class SpectrogramFingerprintData(Data.Dataset):
    def __init__(self, files):
        self.files = files
        self.means = np.load("./datasets/mt/train/data_mean_256.npy")

    def __getitem__(self, index):       
        file = self.files[index]
        org_path = os.path.join("./datasets/mt/train/data", file)
        # org_num = int(file.split("-")[0])
        aug_num = random.randint(1, 9)
        aug_path = org_path[:-6] + f"{aug_num:02d}.npy"
        org_data = np.load(org_path)
        aug_data = np.load(aug_path)
        # org_mean = self.means[org_num - 1]
        start_index, chunk_len = generate_chunk(len(aug_data))
        org_seq, aug_seq = np.zeros((100, 128)), np.zeros((100, 128))
        org_seq[:min(len(org_data), 100)] = org_data[: min(len(org_data), 100)]
        aug_seq[:min(chunk_len, 100)] = aug_data[start_index: min(start_index + chunk_len, start_index + 100)]
        org_mean = np.mean(org_seq[:min(len(org_data), 100)], axis=0)
        aug_mean = np.mean(aug_seq[:min(chunk_len, 100)], axis=0)

        return np.float32(org_seq), np.float32(aug_seq), np.float32(org_mean), np.float32(aug_mean)

    def __len__(self):
        return len(self.files)


class RetrievalData(Data.Dataset):
    def __init__(self, paths, root_dir, max_len):
        self.paths = paths
        self.root_dir = root_dir
        self.max_len = max_len

    def __getitem__(self, index):       
        file_path = self.paths[index]
        data = np.load(self.root_dir + "/" + file_path)
        sequence = np.zeros((100, 128))
        for i in range(0, min(len(data), self.max_len)):
            sequence[i] = data[i]
        # mean = self.means[index]
        mean = np.mean(sequence[:min(len(data), self.max_len)], axis=0)
        return np.float32(sequence), mean

    def __len__(self):
        return len(self.paths)


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
        datefmt="%a %b %d %H:%M:%S %Y",
    )

    work_dir = os.path.join("./log/mt", time.strftime("%Y-%m-%d-%H.%M", time.localtime()))
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
    loss_accumulation = loss_avg = 0
    pbar_train = tqdm(train_data, total=train_nb)

    for step, (org_seq, aug_seq, org_mean, aug_mean) in enumerate(pbar_train):
        org_seq = org_seq.cuda()
        aug_seq = aug_seq.cuda()
        org_mean = org_mean.cuda()
        aug_mean = aug_mean.cuda()
        org_embedding = model(org_seq) + org_mean
        aug_embedding = model(aug_seq) + aug_mean
        loss = loss_info_nce(org_embedding, aug_embedding)
        
        if loss_accumulation is None and loss_avg is None:
            loss_accumulation = loss.item()
            loss_avg = loss.item()
        else:
            loss_accumulation += loss.item()
            loss_avg = loss_accumulation / (step + 1)

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        s = "train ===> epoch:{} ---- step:{} ---- loss:{:.4f} ---- loss_avg:{:.4f}".format(
            epoch, step, loss, loss_avg
        )
        pbar_train.set_description(s)
        logger.info(s)

    return loss_avg


def get_vector_mt(data_source, output_root_dir):
    model.eval()
    db_nb = len(data_source)
    pbar = tqdm(data_source, total=db_nb)

    arr_shape = (len(os.listdir(output_root_dir)), 128)
    arr = np.memmap(output_root_dir +".mm",
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    np.save(output_root_dir + "_shape.npy", arr_shape) 
        
    for i, (vector, mean) in enumerate(pbar):
        emb = model(vector.cuda()).detach().cpu()
        emb = F.normalize(mean + emb)
        arr[i * config.batch_size:(i + 1) * config.batch_size, :] = emb.numpy()
    arr.flush(); del(arr)


def retrieval_new(query_name):
    db_shape = np.load("./database/db_shape.npy")
    db = np.memmap("./database/db.mm", dtype='float32', mode='r+', shape=(db_shape[0], db_shape[1]))
    query_shape = np.load(f"./runs/retrieval/test/{query_name}_shape.npy")
    query = np.memmap(f"./runs/retrieval/test/{query_name}.mm", dtype='float32', mode='r+', shape=(query_shape[0], query_shape[1]))

    db_index = np.load("./database/db_inf.npy")
    db_index_map = dict()
    for i in range(len(db_index)):
        db_index_map[db_index[i][0]] = i

    xb_len = db_shape[0]
    xb = np.zeros((xb_len, 128))
    for i in range(xb_len):
            xb[i] = db[i]

    index = faiss.IndexFlatIP(128)
    index.add(xb)

    correct_num_1 = 0
    correct_num_10= 0
    filenames = os.listdir("./runs/retrieval/test/db")
    for ti in range(query_shape[0]):
        qi = int(filenames[ti][:-4])
        q = np.zeros((1, 128))
        q[0] = query[ti]
        D, I = index.search(q, 10)

        if db_index_map[qi] == I[0][0]:
            correct_num_1 += 1
            correct_num_10 += 1
        elif db_index_map[qi] in I[0]:
            correct_num_10 += 1

    return correct_num_1 / query_shape[0], correct_num_10 / query_shape[0]


def test():
    get_vector_mt(db_data,"./database/db")
    get_vector_mt(query_data, "./runs/retrieval/test/query")
    top1, top10 = retrieval_new("query")
    print("top1 accuracy:", top1, "   ", "top10 accuracy:", top10)
    return top1, top10


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.from_json_file("config.json")
    logger = get_logger()
    val_flag = False
    log_dir = "./runs/checkpoint/mt_infoNCE/exp/mt.pth"

    model = MT(
        input_dim=128,
        output_dim=128,
        dim=256,
        depth=8,
        heads=32,
        dim_head=16,
        mlp_dim=256,
        dropout=0.1,
    ).cuda()

    train_files = load_file_org("./datasets/mt/train/data")
    train_data = SpectrogramFingerprintData(files=train_files)
    train_data = Data.DataLoader(train_data, shuffle=True, batch_size=config.batch_size)

    db_mean = np.load("./database/db_mean_256.npy")
    db_data = RetrievalData(paths=os.listdir("./database/db"), root_dir="./database/db", max_len = 100, means=db_mean)
    db_data = Data.DataLoader(db_data, shuffle=False, batch_size=config.batch_size)

    query_mean = np.load("./runs/retrieval/test/query_mean_256.npy")
    query_data = RetrievalData(paths=os.listdir("./runs/retrieval/test/query"), root_dir="./runs/retrieval/test/query", max_len = 19, means=query_mean)
    query_data = Data.DataLoader(query_data, shuffle=False, batch_size=config.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_info_nce = InfoNCELoss().cuda()

    if val_flag:
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        exit()

    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        print("加载 epoch {} 成功！".format(start_epoch))
    else:
        start_epoch = 0
        print("无保存模型，从头开始训练！")
        df_train = pd.DataFrame(
            columns=[
                "time",
                "epoch",
                "train/loss",
                "test/top1",
                "test/top10",
            ]
        )
        df_train.to_csv("./runs/checkpoint/mt_infoNCE/exp/result.csv", index=False)
        start_epoch = 0

    for epoch in range(start_epoch + 1, config.epochs + 1):
        train_loss = train()
        test_top1, test_top10 = test()
        time_now = time.strftime("%Y-%m-%d-%H.%M", time.localtime())
        data_list = [time_now, epoch, train_loss, test_top1, test_top10]
        data = pd.DataFrame([data_list])
        data.to_csv("./runs/checkpoint/mt_infoNCE/exp/result.csv", mode="a", header=False, index=False)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(state, log_dir)

        if epoch % 10 == 0:
            torch.save(state, "./runs/checkpoint/mt_infoNCE/exp/mt_{}.pth".format(epoch))
