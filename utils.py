import random
import pickle
import numpy as np
from tqdm import tqdm
import os
from thop import profile
import sys
from pathlib import Path
import time
import faiss


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # project's root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def generate_test_ids(max_len, length_list, test_num, save_path):
    test_ids_dict = dict()
    for chunk_length in length_list:
        id_range = max_len - chunk_length
        id_set = set()
        while len(id_set) < min(test_num, max_len - chunk_length + 1):
            id = random.randint(0, id_range)
            id_set.add(id)

        test_ids_dict[chunk_length] = list(id_set)

    with open(save_path, "wb") as file:
        pickle.dump(test_ids_dict, file)

    return test_ids_dict


def generate_chunk(data_len, max_len):
    mid_start = max((data_len - max_len) // 2, 0)
    x = random.random()
    y = x ** 3
    chunk_len = max(int(y * min(data_len, max_len)), 3)
    select_start = max(0, mid_start - chunk_len // 2)
    interval_len = mid_start - select_start
    select_end = mid_start + min(data_len, max_len) - chunk_len + interval_len
    start_index = random.randint(select_start, select_end)
    overlap_start = max(start_index, mid_start)
    overlap_end = min(start_index + chunk_len - 1, mid_start + min(data_len, max_len) - 1)
    overlap_len = overlap_end - overlap_start + 1
    return start_index, chunk_len, overlap_len


def split_database(org_dir, output_dir, max_len):
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    for filename in tqdm(os.listdir(org_dir)):
        path = os.path.join(org_dir, filename)
        data = np.load(path)
        for ri, i in enumerate(range(0, len(data), max_len)):
            chunk_len = min(max_len, len(data) - i)
            if chunk_len != max_len:
                break
            new_path = os.path.join(output_dir, f"{filename[:-4]}_{ri}.npy")
            np.save(new_path, data[i :i+chunk_len])


def eval_model(model, input_data):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params}")
    flops, _ = profile(model, inputs=(input_data))
    print(f"FLOPs: {flops}")


def eval_speed_metric(dummy_db_path, query_db_path, index_type, feature_dim):
    time_cs = time.time()
    dummy_db_shape = np.load(dummy_db_path[:-3] + '_shape.npy')
    dummy_db = np.memmap(
        dummy_db_path,
        dtype='float32',
        mode='r',
        shape=(dummy_db_shape[0], dummy_db_shape[1]),
    )
    query_db_shape = np.load(query_db_path[:-3] + '_shape.npy')
    query_db = np.memmap(
        query_db_path,
        dtype='float32',
        mode='r',
        shape=(query_db_shape[0], query_db_shape[1]),
    ) 
    index = get_index(index_type, dummy_db, dummy_db.shape, 1e7, feature_dim)
    index.add(dummy_db)
    index.add(query_db)
    del query_db, dummy_db
    print(f'create retrieval database using time {time.time() - time_cs} s.')

    query = np.zeros((1, dummy_db_shape[1]))
    time_ss = time.time()
    for _ in range(10):
        D, I = index.search(query, 10)
    print(f'one query using time {(time.time() - time_ss) * 100.} ms.')


def get_index(mode, train_data, train_data_shape, max_nitem_train, feature_dim):
    if mode == 'ip':
        index = faiss.IndexFlatIP(train_data_shape[1])
    elif mode == 'ivfpq':
        index = faiss.IndexFlatL2(train_data_shape[1])
        index = faiss.IndexIVFPQ(index, feature_dim, 256, 64, 8)
        print('Training index using {:>3.2f} % of data...'.format(
            100. * max_nitem_train / len(train_data)))
        sel_tr_idx = np.random.permutation(len(train_data))
        sel_tr_idx = sel_tr_idx[:int(max_nitem_train)]
        index.train(train_data[sel_tr_idx,:])
        index.nprobe = 40
    else:
        raise ValueError(mode.lower())

    return index
