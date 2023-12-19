import random
import pickle


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