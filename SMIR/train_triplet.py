import os
from pathlib import Path
import random
import numpy as np
from config import Config
import torch
from torch import nn
from torch.utils import data as Data
from tqdm import tqdm
from models.mt import MT
from losses.Triplet import TripletLoss
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
import logging
import faiss


def load_file_org(input_file_path):
    files = list()
    for filename in os.listdir(input_file_path):
        if filename.split("-")[1] == "00.npy":
            files.append(filename)
    return files


def generate_chunk(len):
    chunk_len = random.randint(1, len)
    start_index = random.randint(0, len - chunk_len)
    return start_index, chunk_len


class SpectrogramFingerprintData(Data.Dataset):
    def __init__(self, files):
        self.files = files
        self.means = np.load("./datasets/mt/train/data_mean_256.npy")

    def __getitem__(self, index):
        file = self.files[index]
        a_index = int(file.split("-")[0])
        a_num = random.randint(1, 9)

        a_path = f"./datasets/mt/train/data/{a_index}-{a_num:02d}.npy"
        p_path = "./datasets/mt/train/data/" + file
        n_index = a_index
        while n_index == a_index:
            n_index = random.randint(1, 20000)
        n_path = "./datasets/mt/train/data/" + str(n_index) + "-00.npy"

        a_data, p_data, n_data = np.load(a_path), np.load(p_path), np.load(n_path)
        a_start, a_len = generate_chunk(len(a_data))
        a_seq, p_seq, n_seq = np.zeros((100, 128)), np.zeros((100, 128)), np.zeros((100, 128))

        a_seq[:min(a_len, 100)] = a_data[a_start: min(a_start + a_len, a_start + 100)]
        p_seq[:min(len(p_data), 100)] = p_data[: min(len(p_data), 100)]
        n_seq[:min(len(n_data), 100)] = n_data[: min(len(n_data), 100)]
        
        a_mean = np.mean(a_seq[:min(a_len, 100)], axis=0)
        p_mean = np.mean(p_seq[:min(len(p_data), 100)], axis=0)
        n_mean = np.mean(n_seq[:min(len(n_data), 100)], axis=0)

        return np.float32(a_seq), np.float32(p_seq), np.float32(n_seq), np.float32(a_mean), np.float32(p_mean), np.float32(n_mean)

    def __len__(self):
        return len(self.files)


class RetrievalData(Data.Dataset):
    def __init__(self, paths, root_dir, max_len, means):
        self.paths = paths
        self.root_dir = root_dir
        self.max_len = max_len
        self.means = means

    def __getitem__(self, index):       
        file_path = self.paths[index]
        data = np.load(self.root_dir + "/" + file_path)
        sequence = np.zeros((100, 128))
        for i in range(0, min(len(data), self.max_len)):
            sequence[i] = data[i]
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

    work_dir = os.path.join("./log/mit", time.strftime("%Y-%m-%d-%H.%M", time.localtime()))
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

    for step, (a_x, p_x, n_x, a_mean, p_mean, n_mean) in enumerate(pbar_train):
        a_x = a_x.cuda()
        p_x = p_x.cuda()
        n_x = n_x.cuda()
        a_mean = a_mean.cuda()
        p_mean = p_mean.cuda()
        n_mean = n_mean.cuda()
        a_out = model(a_x) + a_mean
        p_out = model(p_x) + p_mean
        n_out = model(n_x) + n_mean
        s_d = F.cosine_similarity(a_out, p_out)
        n_d = F.cosine_similarity(a_out, n_out)
        thing1 = (s_d - n_d < config.alpha).flatten().cpu()
        thing2 = (s_d - n_d >= config.alpha).flatten().cpu()
        mask = np.where(thing1.numpy() == 1)[0]
        correct_num += torch.sum(thing2).item()
        if not len(mask):
            continue
        a_out, p_out, n_out = a_out[mask], p_out[mask], n_out[mask]
        loss = torch.mean(loss_t_fc(a_out, p_out, n_out))

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

    accuracy_train = correct_num / total_num_train
    
    return loss_avg, accuracy_train


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
    log_dir = "./runs/checkpoint/mt_triplet/exp/mt.pth"

    model = MT(
        input_dim=128,
        output_dim=128,
        dim=256,
        depth=6,
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
    loss_t_fc = TripletLoss(config.alpha)

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
                "train/accuracy",
                "test/top1",
                "test/top10",
            ]
        )
        df_train.to_csv("./runs/checkpoint/mt_triplet/exp/result.csv", index=False)
        start_epoch = 0

    for epoch in range(start_epoch + 1, config.epochs + 1):
        train_loss, train_acc = train()
        test_top1, test_top10 = test()

        time_now = time.strftime("%Y-%m-%d-%H.%M", time.localtime())
        data_list = [time_now, epoch, train_loss,train_acc, test_top1, test_top10]
        data = pd.DataFrame([data_list])
        data.to_csv("./runs/checkpoint/mt_triplet/exp/result.csv", mode="a", header=False, index=False)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(state, log_dir)

        if epoch % 10 == 0:
            torch.save(state, "./runs/checkpoint/mt_triplet/exp/mt_{}.pth".format(epoch))


# def val():
#     model.eval()
#     total_num_val = len(val_data.dataset)
#     loss_accumulation = 0
#     loss_avg = 0
#     correct_num = 0
#     pbar_val = tqdm(val_data, total=len(val_data))

#     with torch.no_grad():
#         for step, (a_x, p_x, n_x) in enumerate(pbar_val):
#             a_x = a_x.cuda()
#             p_x = p_x.cuda()
#             n_x = n_x.cuda()

#             a_out, p_out, n_out = model(a_x), model(p_x), model(n_x)

#             s_d = F.pairwise_distance(a_out, p_out)
#             n_d = F.pairwise_distance(a_out, n_out)

#             thing1 = (n_d - s_d < config.alpha).flatten().cpu()
#             thing2 = (n_d - s_d >= config.alpha).flatten().cpu()

#             mask = np.where(thing1.numpy() == 1)[0]
#             correct_num += torch.sum(thing2).item()

#             if not len(mask):
#                 continue

#             a_out, p_out, n_out = a_out[mask], p_out[mask], n_out[mask]
#             loss = torch.mean(loss_t_fc(a_out, p_out, n_out))

#             if loss_accumulation is None and loss_avg is None:
#                 loss_accumulation = loss.item()
#                 loss_avg = loss.item()
#             else:
#                 loss_accumulation += loss.item()
#                 loss_avg = loss_accumulation / (step + 1)

#     triple_accuracy_val = correct_num / total_num_val

#     s = "val ===> epoch:{}  ---- loss_avg:{:.4f} ---- triple_accuracy:{:.4f}".format(
#         epoch, loss_avg, triple_accuracy_val
#     )
#     logger.info(s)

#     return loss_avg, triple_accuracy_val