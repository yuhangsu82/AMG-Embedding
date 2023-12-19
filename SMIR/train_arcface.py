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
from models.mt_test import MT1, MT2
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm
import logging
from losses.ArcFace import ArcFaceLoss, ArcFaceLoss_new
import argparse
import faiss
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses.Tam import TamLoss_1
import pickle


def init_model(model_type):
    if model_type == 'mt':
        # model = MT(
        #     input_dim=128,
        #     output_dim=128,
        #     dim=256,
        #     depth=8,
        #     heads=32,
        #     dim_head=16,
        #     mlp_dim=256,
        #     dropout=0.2,
        # ).cuda()
        model = MT(
            input_dim=128,
            output_dim=opt.feature_dim,
            dim=256,
            depth=16,
            heads=64,
            dropout=0.1,
        ).cuda()
    elif model_type == 'mt_test':
        model = MT1(
            input_dim=128,
            output_dim=opt.feature_dim,
            dim=256,
            depth=10,
            heads=8,
            dropout=0.1,
            max_len=64,
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


def load_file_org(input_file_path):
    files = list()
    for filename in os.listdir(input_file_path):
        if filename.split("-")[1] == "00.npy":
            files.append(filename)
    return files


def generate_chunk(len):
    x = random.random()
    y = x ** 3
    chunk_len = min(max(int(y * len), 1), len)
    start_index = random.randint(0, len - chunk_len)
    return start_index, chunk_len


class SpectrogramFingerprintData(Data.Dataset):
    def __init__(self, files, data_path):
        self.files = files
        self.data_path = data_path

    def __getitem__(self, index):
        file = self.files[index]
        path = os.path.join(self.data_path, file)
        data = np.load(path)
        label = int(file.split("-")[0]) - 1
    
        length = min(len(data), 64)
        start_index, chunk_len = generate_chunk(length)
        if int(file.split("-")[1].split(".")[0]) == 0:
            start_index = 0
            chunk_len = len(data)
        sample_weight = 1
        margin = 0.2 * chunk_len / len(data) + 0.2
        sequence = np.zeros((64, 128), np.float32)
        padding_mask = np.zeros((64), dtype=np.bool_)   
        padding_mask[chunk_len:] = 1
        sequence[:min(chunk_len, 64)] = data[start_index: min(start_index + chunk_len, start_index + 64)]

        return np.float32(sequence), np.int64(label), np.float32(margin), np.float32(sample_weight), np.bool_(padding_mask)

    def __len__(self):
        return len(self.files)


class RetrievalData(Data.Dataset):
    def __init__(self, paths, root_dir, max_len=64, start_id=0):
        self.paths = paths
        self.root_dir = root_dir
        self.max_len = max_len
        self.start_id = start_id

    def __getitem__(self, index):       
        file_path = self.paths[index]
        data = np.load(self.root_dir + "/" + file_path)

        sequence = np.zeros((64, 128), dtype=np.float32)
        length = min(len(data), 64)

        if self.max_len == 64:
            chunk_len = length
        else:
            chunk_len = self.max_len
        
        for i in range(min(length, self.max_len)):
            sequence[i] = data[i + self.start_id]

        padding_mask = np.zeros((64), dtype=np.bool_)
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
    for step, (tensor, label, margin, sample_weight, padding_mask) in enumerate(pbar_train):
        tensor = tensor.cuda()
        label = label.cuda()
        margin = margin.cuda()
        sample_weight = sample_weight.cuda()   
        padding_mask = padding_mask.cuda()
        
        local_embedding = model(tensor, padding_mask)
        # local_embedding = model(tensor)
        logit, loss = arcface(local_embedding, label, margin, sample_weight)   
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


def init_arcface_center_256():
    center = np.zeros((opt.class_nums, 256), dtype=np.float32)
    for i in range(1, opt.class_nums + 1):
        data = np.load(opt.dataset_path + f"/{i}-00.npy")
        center[i-1][:128] = np.mean(data, axis=0)
        center[i-1][128:] = np.mean(data, axis=0)
    t_c = torch.from_numpy(center).float().cuda()
    with torch.no_grad():
        arcface.weight.data = t_c


def get_vector_mt(data_source, output_root_dir, mode="none"):
    model.eval()
    db_nb = len(data_source)
    if mode == "db":
        pbar = tqdm(data_source, total=db_nb, disable=False)
    else:
        pbar = tqdm(data_source, total=db_nb, disable=True)

    arr_shape = (len(os.listdir(output_root_dir)), opt.feature_dim)
    arr = np.memmap(output_root_dir +".mm",
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    np.save(output_root_dir + "_shape.npy", arr_shape)       
    for i, (vector, padding_mask) in enumerate(pbar):
        emb = F.normalize(model(vector.cuda(), padding_mask.cuda())).detach().cpu()
        # emb = F.normalize(model(vector.cuda())).detach().cpu()
        arr[i * 400:(i + 1) * 400, :] = emb.numpy()
    arr.flush(); del(arr)


def retrieval(query_name):
    db_shape = np.load("./database/db_shape.npy")
    db = np.memmap("./database/db.mm", dtype='float32', mode='r+', shape=(db_shape[0], db_shape[1]))
    query_shape = np.load(f"./runs/retrieval/test/{query_name}_shape.npy")
    query = np.memmap(f"./runs/retrieval/test/{query_name}.mm", dtype='float32', mode='r+', shape=(query_shape[0], query_shape[1]))
    db_index = np.load("./database/db_inf.npy")
    db_index_map = dict()
    for i in range(len(db_index)):
        db_index_map[db_index[i][0]] = i
    xb_len = db_shape[0]
    xb = np.zeros((xb_len, opt.feature_dim) ,dtype=np.float32)
    for i in range(xb_len):
            xb[i] = db[i]
    index = faiss.IndexFlatIP(opt.feature_dim)
    index.add(xb)
    correct_num_1 = 0
    correct_num_10= 0
    filenames = os.listdir("./runs/retrieval/test/db")
    for ti in range(query_shape[0]):
        qi = int(filenames[ti][:-4])
        q = np.zeros((1, opt.feature_dim), np.float32)
        q[0] = query[ti]
        D, I = index.search(q, 10)

        if db_index_map[qi] == I[0][0]:
            correct_num_1 += 1
            correct_num_10 += 1
        elif db_index_map[qi] in I[0]:
            correct_num_10 += 1

    return correct_num_1 / query_shape[0], correct_num_10 / query_shape[0]


def test():
    db_data = RetrievalData(paths=os.listdir("./database/db"), root_dir="./database/db", max_len=64, start_id=0)
    db_data = Data.DataLoader(db_data, shuffle=False, batch_size=400)
    get_vector_mt(db_data, "./database/db", "db")
    test_ids_dict = pickle.load(open("./runs/retrieval/test/test_ids.pickle", "rb"))
    top1 = list()
    top10 = list()
    for seg_len in tqdm(test_ids_dict.keys()):
        top1_scores = list()
        top10_scores = list()
        for id in test_ids_dict[seg_len]:
            query_data = RetrievalData(paths=os.listdir("./runs/retrieval/test/query"), root_dir="./runs/retrieval/test/query", max_len=seg_len, start_id=id)
            query_data = Data.DataLoader(query_data, shuffle=False, batch_size=400)
            get_vector_mt(query_data, "./runs/retrieval/test/query")
            seg_top1, seg_top10 = retrieval("query")
            top1_scores.append(seg_top1)
            top10_scores.append(seg_top10)
        
        top1.append(sum(top1_scores)/len(top1_scores))
        top10.append(sum(top10_scores)/len(top10_scores))

    return top1, top10


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='./datasets/mt/train/data')
    parser.add_argument("--model", type=str, default='mt_test')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--last_lr", type=float, default=0.00001)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--class_nums", type=int, default=20000)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    return parser.parse_args()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt = parse_opt()
    model = init_model(opt.model)
    logger = get_logger()
    # log_dir = f"./runs/checkpoint/{opt.model}_arcface/exp/{opt.model}.pth"
    log_dir = "./runs/checkpoint/mt_test_arcface/exp-2w-256/mt_test_6.pth"

    train_files = load_file(opt.dataset_path)

    train_data = SpectrogramFingerprintData(files=train_files, data_path=opt.dataset_path)
    train_data = Data.DataLoader(train_data, shuffle=True, batch_size=opt.batch_size)
    arcface = TamLoss_1(in_features=opt.feature_dim, out_features=opt.class_nums, s=64.0).cuda()
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': arcface.parameters()}], lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch * len(train_data), eta_min=opt.last_lr)

    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint["model"])
        arcface.load_state_dict(checkpoint["arcface"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
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
            ]
        )
        df_train.to_csv(f"./runs/checkpoint/{opt.model}_arcface/exp/result.csv", index=False)
        start_epoch = 0


    for epoch in range(start_epoch + 1, opt.epoch + 1):
        if epoch == 1:
            init_arcface_center_256()

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
        if epoch % 10 == 0:
            print("epoch: ", epoch)
            table = PrettyTable()
            table.field_names = ["segments", "1", "3", "5", "9", "11", "19", "39"]
            top1, top10 = test()
            table.add_row(["Top1"] + top1)
            table.add_row(["Top10"] + top10)
            table.align = "r"
            table_string = table.get_string()
            print(table)

            with open(f"./runs/checkpoint/{opt.model}_arcface/exp/table.txt", "a") as file:
                file.write(f"epoch: {epoch}\n")
                file.write(table_string + "\n")

            
            
        torch.save(state, f"./runs/checkpoint/{opt.model}_arcface/exp/{opt.model}_{epoch}.pth".format(epoch))
        torch.save(state, log_dir)
        
