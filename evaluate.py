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


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # project's root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def init_model(model_type, feature_dim, is_residual, device):
    if model_type == 'mt':
        model = MT(128, feature_dim, 256, 10, 8, 0.1, is_residual).to(device)
    elif model_type == 'mlstm':
        model = MLSTM(128, feature_dim, 256, 8, 0.1, is_residual).to(device)
    elif model_type == 'mcnn':
        model = MCNN(feature_dim, is_residual).to(device)
    else:
        raise Exception('No such model! (Tip: You can fix the code and add your own model in this project.)')

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
    emb_dummy_dir, emb_dir, max_len, batch_size, feature_dim, device, model
):
    db_data = SpectrogramFingerprintData(
        files=os.listdir(emb_dummy_dir),
        root_dir=emb_dummy_dir,
        max_len=max_len,
        seq_len=max_len,
        start_id=0,
        feature_dim=feature_dim,
        mode='test',
    )
    db_data = Data.DataLoader(db_data, shuffle=False, batch_size=batch_size)
    dummy_db = generate_feature(
        db_data, emb_dummy_dir, feature_dim, device, batch_size, model, 'db'
    )
    dummy_db_shape = np.load(str(emb_dummy_dir) + '_shape.npy')

    query_db_data = SpectrogramFingerprintData(
        files=os.listdir(os.path.join(emb_dir, 'db')),
        root_dir=os.path.join(emb_dir, 'db'),
        max_len=max_len,
        seq_len=max_len,
        start_id=0,
        feature_dim=feature_dim,
        mode='test',
    )
    query_db_data = Data.DataLoader(query_db_data, shuffle=False, batch_size=batch_size)
    query_db = generate_feature(
        query_db_data,
        os.path.join(emb_dir, 'db'),
        feature_dim,
        device,
        batch_size,
        model,
        'db',
    )
    query_db_shape = np.load(os.path.join(emb_dir, 'db') + '_shape.npy')

    fake_recon_index = np.memmap(
        ROOT / 'database/merge_db.mm',
        dtype='float32',
        mode='w+',
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
    os.remove(ROOT / 'database/merge_db.mm')
    faiss.write_index(index, str(emb_dummy_dir) + '.index')
    # index = faiss.read_index(emb_dummy_dir.as_posix() +'.index')

    return dummy_db_shape, query_db_shape, index


def eval(
    emb_dir,
    emb_dummy_dir,
    model_type,
    is_residual,
    checkpoint_path,
    max_len,
    device,
    batch_size,
    feature_dim,
    k_prob,
):
    model = init_model(model_type, feature_dim, is_residual, device)
    if torch.cuda.is_available() and 'cuda' in device:
        device = torch.device(device)
    else:
        device = torch.device('cpu')

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    dummy_db_shape, query_shape, index = create_index(
        emb_dummy_dir, emb_dir, max_len, batch_size, feature_dim, device, model
    )
    test_ids_dict = pickle.load(open(os.path.join(emb_dir, 'test_ids.pickle'), 'rb'))
    top1 = list()
    top10 = list()
    id_nums = len(test_ids_dict[next(iter(test_ids_dict))])
    pbar = tqdm(
        total=len(test_ids_dict.keys()) * id_nums * query_shape[0], desc='Searching'
    )
    table = PrettyTable()
    test_seq_len = list(test_ids_dict.keys())
    table.field_names = ['segments'] + test_seq_len

    for seg_len in test_seq_len:
        top1_correct = list()
        top10_correct = list()
        for id in test_ids_dict[seg_len]:
            query_data = SpectrogramFingerprintData(
                files=os.listdir(os.path.join(emb_dir, 'tr-aug')),
                root_dir=os.path.join(emb_dir, 'tr-aug'),
                max_len=max_len,
                seq_len=seg_len,
                start_id=id,
                feature_dim=feature_dim,
                mode='test',
            )
            query_data = Data.DataLoader(
                query_data, shuffle=False, batch_size=batch_size
            )
            query = generate_feature(
                query_data,
                os.path.join(emb_dir, 'tr-aug'),
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
    table.add_row(['Top1'] + top1)
    table.add_row(['Top10'] + top10)
    table.align = 'r'
    print(table)

    return table.get_string()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=ROOT / 'runs/checkpoint/mt_tam/exp-2w-s=32/mt_test_50.pth')
    parser.add_argument('--emb_dir', type=str, default=ROOT / 'runs/retrieval/test_15')
    parser.add_argument('--emb_dummy_dir', type=str, default=ROOT / 'database/fma_full_15s')
    parser.add_argument('--model_type', type=str, default='mt')
    parser.add_argument('--is_residual', type=bool, default=False)
    parser.add_argument('--max_len', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=2400)
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--k_prob', type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()

    table_string = eval(
        emb_dir=opt.emb_dir,
        emb_dummy_dir=opt.emb_dummy_dir,
        model_type=opt.model_type,
        is_residual=opt.is_residual,
        checkpoint_path=opt.checkpoint_path,
        max_len=opt.max_len,
        device=opt.device,
        batch_size=opt.batch_size,
        feature_dim=opt.feature_dim,
        k_prob=opt.k_prob,
    )
