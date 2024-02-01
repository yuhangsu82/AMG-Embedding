'''
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
'''


import argparse
import os
import sys
from pathlib import Path
import numpy as np
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
import pickle
from dataset import SpectrogramFingerprintData
from models import MCNN, MLSTM, MT
import math
from collections import defaultdict


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # project's root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def init_model(model_type, feature_dim, device):
    if model_type == 'mt':
        model = MT(feature_dim, feature_dim, 256, 10, 8, 0.1).to(device)
    elif model_type == 'mlstm':
        model = MLSTM(feature_dim, feature_dim, 256, 8, 0.1).to(device)
    elif model_type == 'mcnn':
        model = MCNN(feature_dim).to(device)

    return model


def generate_feature(
    data_source, output_root_dir, feature_dim, device, batch_size, model, mode='none'
):
    model.eval()
    db_nb = len(data_source)
    if mode == 'db':
        pbar = tqdm(data_source, total=db_nb, disable=False)
    else:
        pbar = tqdm(data_source, total=db_nb, disable=True)

    arr_shape = (len(os.listdir(output_root_dir)), feature_dim)
    arr = np.memmap(
        str(output_root_dir) + '.mm', dtype='float32', mode='w+', shape=arr_shape
    )
    np.save(str(output_root_dir) + '_shape.npy', arr_shape)
    for i, (vector, padding_mask) in enumerate(pbar):
        emb = model(vector.to(device), padding_mask.to(device)).detach().cpu()
        arr[i * batch_size : (i + 1) * batch_size, :] = emb.numpy()
    arr.flush()
    return arr


def create_index(
    emb_dummy_dir, emb_dir, max_len, duration_max, batch_size, feature_dim, device, model
):
    db_data = SpectrogramFingerprintData(
        files=os.listdir(emb_dummy_dir),
        root_dir=emb_dummy_dir,
        max_len=max_len,
        seq_len=2 * duration_max - 1,
        start_id=[0 for _ in range(len(os.listdir(emb_dummy_dir)))],
        feature_dim=feature_dim,
        mode='test',
    )
    db_data = Data.DataLoader(db_data, shuffle=False, batch_size=batch_size)

    dummy_db = generate_feature(
        db_data, emb_dummy_dir, feature_dim, device, batch_size, model, 'db'
    )
    dummy_db_shape = np.load(str(emb_dummy_dir) + '_shape.npy')
    # dummy_db = np.memmap(
    #     str(emb_dummy_dir) + '.mm',
    #     dtype='float32',
    #     mode='r',
    #     shape=(dummy_db_shape[0], feature_dim),
    # )

    query_db_data = SpectrogramFingerprintData(
        files=os.listdir(os.path.join(emb_dir, f'db/test_{duration_max}s')),
        root_dir=os.path.join(emb_dir, f'db/test_{duration_max}s'),
        max_len=max_len,
        seq_len=2 * duration_max - 1,
        start_id=[0 for _ in range(len(os.listdir(os.path.join(emb_dir, f'db/test_{duration_max}s'))))],
        feature_dim=feature_dim,
        mode='test',
    )
    query_db_data = Data.DataLoader(query_db_data, shuffle=False, batch_size=batch_size)
    query_db = generate_feature(
        query_db_data,
        os.path.join(emb_dir, f'db/test_{duration_max}s'),
        feature_dim,
        device,
        batch_size,
        model,
        'db',
    )
    index = faiss.IndexFlatIP(feature_dim)
    index.add(dummy_db)
    index.add(query_db)
    del query_db, dummy_db

    return dummy_db_shape, index


def eval(
    emb_dir,
    emb_dummy_dir,
    model_type,
    checkpoint_path,
    duration_max,
    device,
    batch_size,
    feature_dim,
    k_prob,
):
    model = init_model(model_type, feature_dim, device)
    if torch.cuda.is_available() and 'cuda' in device:
        device = torch.device(device)
    else:
        device = torch.device('cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    duration_map = {5: 12, 10: 24, 15: 32, 30: 64}
    max_len = duration_map[duration_max]
    top1 = list()
    top10 = list()
    dummy_db_shape, index = create_index(emb_dummy_dir, emb_dir, max_len, duration_max, batch_size, feature_dim, device, model)
    with open(ROOT / 'runs/retrieval/test_ids_sigir2024.pkl', 'rb') as file:
        test_data = pickle.load(file)
    test_ids_dict = test_data[duration_max]
    test_seq_len = list(test_ids_dict.keys())
    seconds_len = 2 * duration_max - 1

    for seg_len in tqdm(test_seq_len, desc="Searching"):
        query_dir = os.path.join(emb_dir, 'query')
        query_files = os.listdir(query_dir)
        correct_num_1 = 0
        correct_num_10 = 0
        for id in test_ids_dict[seg_len]:
            scores = [defaultdict(int) for _ in range(len(query_files))]
            for si in range(math.ceil(seg_len/seconds_len)):
                if si == seg_len//seconds_len:
                    query_len = seg_len%seconds_len
                else:
                    query_len = seconds_len
                if query_len < 3:
                    continue
                query_data = SpectrogramFingerprintData(
                    files=query_files,
                    root_dir=query_dir,
                    max_len=max_len,
                    seq_len=query_len,
                    start_id=[id + si * seconds_len for _ in range(len(query_files))],
                    feature_dim=feature_dim,
                    mode='test',
                )
                query_data = Data.DataLoader(query_data, shuffle=False, batch_size=batch_size)
                query = generate_feature(
                    query_data,
                    query_dir,
                    feature_dim,
                    device,
                    batch_size,
                    model,
                )
                D, I = index.search(query, k_prob)
                for ti in range(len(query_files)):
                    for ki in range(k_prob):
                        idx = I[ti][ki] - si
                        if idx >= 0:
                            scores[ti][idx] += D[ti][ki]
                del query

            for qi in range(len(query_files)):
                get_id = qi * (math.floor(59 / seconds_len)) + (math.floor(id / seconds_len)) + dummy_db_shape[0]
                top_10_ids = sorted(scores[qi], key=scores[qi].get, reverse=True)[:10]
                if get_id == top_10_ids[0] or get_id + 1 == top_10_ids[0]:
                    correct_num_1 += 1
                    correct_num_10 += 1
                elif get_id in top_10_ids or get_id + 1 in top_10_ids:
                    correct_num_10 += 1
        
        top1.append(100.0 * correct_num_1 / len(query_files) / len(test_ids_dict[seg_len]))
        top10.append(100.0 * correct_num_10 / len(query_files) / len(test_ids_dict[seg_len]))

    table = PrettyTable()
    table.field_names = ['segments'] + test_seq_len
    table.add_row(['Top1'] + top1)
    table.add_row(['Top10'] + top10)
    table.align = 'r'
    print(table)

    return table.get_string()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=ROOT / 'runs/checkpoint/mt_pam/exp-10s/mt_50.pth', help='Path to the model checkpoint to be loaded')
    parser.add_argument('--emb_dir', type=str, default=ROOT / 'runs/retrieval', help='Directory for the embeddings')
    parser.add_argument('--emb_dummy_dir', type=str, default=ROOT / 'database/fma_full_10s', help='Directory for the dummy embeddings')
    parser.add_argument('--model_type', choices=['mcnn', 'mlstm', 'mt'], default='mt', help='Type of the model to be used')
    parser.add_argument('--duration_max', choices=[5, 10, 15, 30], default=10, help='Maximum duration of the audio clips')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to be used for computation (e.g., cuda:0)')
    parser.add_argument('--batch_size', type=int, default=1200, help='Batch size for processing')
    parser.add_argument('--feature_dim', type=int, default=128, help='Dimension of the feature vectors')
    parser.add_argument('--k_prob', type=int, default=10, help='Parameter for the K-probability')

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()

    table_string = eval(
        emb_dir=opt.emb_dir,
        emb_dummy_dir=opt.emb_dummy_dir,
        model_type=opt.model_type,
        checkpoint_path=opt.checkpoint_path,
        duration_max=opt.duration_max,
        device=opt.device,
        batch_size=opt.batch_size,
        feature_dim=opt.feature_dim,
        k_prob=opt.k_prob,
    )