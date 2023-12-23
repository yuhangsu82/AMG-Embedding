import random
import pickle
import numpy as np
from tqdm import tqdm
import os


def generate_test_ids(max_len, length_list, test_num, save_path):
    test_ids_dict = dict()
    for chunk_length in length_list:
        id_range = max_len - chunk_length
        id_set = set()
        while len(id_set) < test_num:
            id = random.randint(0, id_range)
            id_set.add(id)

        test_ids_dict[chunk_length] = list(id_set)

    with open(save_path, "wb") as file:
        pickle.dump(test_ids_dict, file)


def generate_chunk(len):
    x = random.random()
    y = x ** 3
    chunk_len = min(max(int(y * len), 1), len)
    start_index = random.randint(0, len - chunk_len)
    return start_index, chunk_len


def split_retrieval_database(org_dir, output_dir, max_len):
    for filename in tqdm(os.listdir(org_dir)):
        path = os.path.join(org_dir, filename)
        data = np.load(path)
        for ri, i in enumerate(range(0, len(data), max_len)):
            chunk_len = min(max_len, len(data)-i)
            if chunk_len != max_len:
                break
            new_path = os.path.join(output_dir, f"{filename[:-4]}_{ri}.npy")
            np.save(new_path, data[i:i+chunk_len])


# split_retrieval_database("./database/fma_full", "./database/fma_full_10s", 19)
generate_test_ids(19, [1, 3, 5, 9, 11], 5, "./runs/retrieval/test_10/test_ids.pickle")