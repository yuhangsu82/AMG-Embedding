"""
This tip describes how to create a search database and complete the search.
 ----------------------------------------------------------------------
    FAISS index setup

        dummy: 10 items.
        db: 5 items.
        query: 5 items, corresponding to 'db'.

        index.add(dummy_db); index.add(db) # 'dummy_db' first

               |------ dummy_db ------|
        index: [d0, d1, d2,..., d8, d9, d11, d12, d13, d14, d15]
                                       |--------- db ----------|

                                       |--------query ---------|
                                       [q0,  q1,  q2,  q3,  q4]

    • The set of ground truth IDs for q[i] will be (i + len(dummy_db))

    ---------------------------------------------------------------------- 
"""


import argparse
import os
import sys
from pathlib import Path
import numpy as np
from config import Config
import torch
from torch import nn
from torch.utils import data as Data
from tqdm import tqdm
from models.mt import MT
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import faiss
from prettytable import PrettyTable
from models.mt_test import MT1
import pickle


# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # SMIR root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.realpath(ROOT, Path.cwd()))  # relative


def init_model(model_type, feature_dim, device):
    if model_type == "mt":
        model = MT(
            input_dim=128,
            output_dim=128,
            dim=256,
            depth=10,
            heads=8,
            dim_head=16,
            mlp_dim=256,
            dropout=0.1,
        ).to(device)
    elif model_type == "mt_test":
        model = MT1(
            input_dim=128,
            output_dim=feature_dim,
            dim=256,
            depth=10,
            heads=8,
            dropout=0.1,
        ).to(device)
    elif model_type == "mlstm":
        model = MLSTM(128, feature_dim, 256, 8, True, 0.1).to(device)
    elif model_type == "mcnn":
        model = MCNN(feature_dim).to(device)
    else:
        raise Exception("No such model!")

    return model


class RetrievalData(Data.Dataset):
    def __init__(
        self, files, root_dir, max_len=32, seq_len=32, start_id=0, feature_dim=128
    ):
        self.files = files
        self.root_dir = root_dir
        self.max_len = max_len
        self.seq_len = seq_len
        self.start_id = start_id
        self.feature_dim = feature_dim

    def __getitem__(self, index):
        file = self.files[index]
        path = os.path.join(self.root_dir, file)
        data = np.load(path)
        sequence = np.zeros((self.max_len, self.feature_dim), dtype=np.float32)
        length = min(len(data), self.seq_len)
        if self.seq_len == self.max_len:
            chunk_len = length
        else:
            chunk_len = self.seq_len
        sequence[:min(length, self.seq_len)] = data[self.start_id: self.start_id + min(length, self.seq_len)]
        padding_mask = np.zeros((self.max_len), dtype=np.bool_)
        padding_mask[chunk_len:] = 1

        return np.float32(sequence), np.bool_(padding_mask)

    def __len__(self):
        return len(self.files)


def generate_feature(
    data_source, output_root_dir, feature_dim, device, batch_size, model, mode="none"
):
    model.eval()
    db_nb = len(data_source)
    if mode == "db":
        pbar = tqdm(data_source, total=db_nb, disable=False)
    else:
        pbar = tqdm(data_source, total=db_nb, disable=True)

    arr_shape = (len(os.listdir(output_root_dir)), feature_dim)
    arr = np.memmap(
        output_root_dir + ".mm", dtype="float32", mode="w+", shape=arr_shape
    )
    np.save(output_root_dir + "_shape.npy", arr_shape)
    for i, (vector, padding_mask) in enumerate(pbar):
        emb = model(vector.to(device), padding_mask.to(device)).detach().cpu()
        arr[i * batch_size : (i + 1) * batch_size, :] = emb.numpy()
    arr.flush()
    return arr


def create_index(
    emb_dummy_dir, emb_dir, max_len, batch_size, feature_dim, device, model
):
    db_data = RetrievalData(
        files=os.listdir(emb_dummy_dir),
        root_dir=emb_dummy_dir,
        max_len=max_len,
        seq_len=max_len,
        start_id=0,
    )
    db_data = Data.DataLoader(db_data, shuffle=False, batch_size=batch_size)
    dummy_db = generate_feature(
        db_data, emb_dummy_dir, feature_dim, device, batch_size, model, "db"
    )
    dummy_db_shape = np.load(emb_dummy_dir + "_shape.npy")

    query_db_data = RetrievalData(
        files=os.listdir(os.path.join(emb_dir, "db")),
        root_dir=os.path.join(emb_dir, "db"),
        max_len=max_len,
        seq_len=max_len,
        start_id=0,
    )
    query_db_data = Data.DataLoader(query_db_data, shuffle=False, batch_size=batch_size)
    query_db = generate_feature(
        query_db_data,
        os.path.join(emb_dir, "db"),
        feature_dim,
        device,
        batch_size,
        model,
        "db",
    )
    query_db_shape = np.load(os.path.join(emb_dir, "db") + "_shape.npy")

    fake_recon_index = np.memmap(
        "./database/merge_db.mm",
        dtype="float32",
        mode="w+",
        shape=(dummy_db_shape[0] + query_db_shape[0], feature_dim),
    )
    fake_recon_index[: dummy_db_shape[0], :] = dummy_db[:, :]
    fake_recon_index[
        dummy_db_shape[0] : dummy_db_shape[0] + query_db_shape[0], :
    ] = query_db[:, :]
    index = faiss.IndexFlatIP(feature_dim)
    index.add(fake_recon_index)
    fake_recon_index.flush()
    del fake_recon_index, query_db, dummy_db
    os.remove("./database/merge_db.mm")
    faiss.write_index(index, emb_dummy_dir + ".index")
    # index = faiss.read_index(emb_dummy_dir +'.index')

    return dummy_db_shape, query_db_shape, index


def eval(
    emb_dir,
    emb_dummy_dir,
    model_type,
    checkpoint_path,
    max_len,
    device,
    batch_size,
    feature_dim,
    k_prob,
):
    model = init_model(model_type, feature_dim, device)
    if torch.cuda.is_available() and "cuda" in device:
        device = torch.device(device)
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])

    dummy_db_shape, query_shape, index = create_index(
        emb_dummy_dir, emb_dir, max_len, batch_size, feature_dim, device, model
    )
    test_ids_dict = pickle.load(open(os.path.join(emb_dir, "test_ids.pickle"), "rb"))
    top1 = list()
    top10 = list()
    id_nums = len(test_ids_dict[next(iter(test_ids_dict))])
    pbar = tqdm(
        total=len(test_ids_dict.keys()) * id_nums * query_shape[0], desc="Searching"
    )
    table = PrettyTable()
    test_seq_len = list(test_ids_dict.keys())
    table.field_names = ["segments"] + test_seq_len

    for seg_len in test_seq_len:
        top1_correct = list()
        top10_correct = list()
        for id in test_ids_dict[seg_len]:
            query_data = RetrievalData(
                files=os.listdir(os.path.join(emb_dir, "ts-aug")),
                root_dir=os.path.join(emb_dir, "ts-aug"),
                max_len=max_len,
                seq_len=seg_len,
                start_id=id,
            )
            query_data = Data.DataLoader(
                query_data, shuffle=False, batch_size=batch_size
            )
            query = generate_feature(
                query_data,
                os.path.join(emb_dir, "ts-aug"),
                feature_dim,
                device,
                batch_size,
                model,
            )
            correct_num_1 = 0
            correct_num_10 = 0
            for ti in range(query_shape[0]):
                get_id = ti + dummy_db_shape[0]
                q = np.zeros((1, feature_dim))
                q[0] = query[ti]
                D, I = index.search(q, k_prob)
                if get_id == I[0][0]:
                    correct_num_1 += 1
                    correct_num_10 += 1
                elif get_id in I[0]:
                    correct_num_10 += 1

                pbar.update(1)

            top1_correct.append(correct_num_1)
            top10_correct.append(correct_num_10)
            del query

        top1.append(100.0 * sum(top1_correct) / (len(top1_correct) * query_shape[0]))
        top10.append(100.0 * sum(top10_correct) / (len(top10_correct) * query_shape[0]))

    pbar.close()
    table.add_row(["Top1"] + top1)
    table.add_row(["Top10"] + top10)
    table.align = "r"
    print(table)

    return table.get_string()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="./runs/checkpoint/mt_test_arcface/exp/mt_test_20.pth")
    parser.add_argument("--emb_dir", type=str, default="./runs/retrieval/test_15")
    parser.add_argument("--emb_dummy_dir", type=str, default="./database/fma_full_15s")
    parser.add_argument("--model_type", type=str, default="mt_test")
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=2400)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--k_prob", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()

    table_string = eval(
        emb_dir=opt.emb_dir,
        emb_dummy_dir=opt.emb_dummy_dir,
        model_type=opt.model_type,
        checkpoint_path=opt.checkpoint_path,
        max_len=opt.max_len,
        device=opt.device,
        batch_size=opt.batch_size,
        feature_dim=opt.feature_dim,
        k_prob=opt.k_prob,
    )
