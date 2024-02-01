from torch.utils import data as Data
import numpy as np
import os
import random
from utils import generate_chunk


class SpectrogramFingerprintData(Data.Dataset):
    def __init__(self, files, root_dir, max_len=32, seq_len=32, start_id=None, feature_dim=128, class_nums=0, mode='test', mu=None, ml=None):
        self.files = files
        self.root_dir = root_dir
        self.max_len = max_len
        self.seq_len = seq_len
        self.start_id = start_id
        self.feature_dim = feature_dim
        self.class_nums = class_nums
        self.mode = mode
        self.mu = mu
        self.ml = ml

    def __getitem__(self, index):
        if self.mode == 'train-pam':
            return self.get_item_pam(index)
        elif self.mode == 'train-arcface':
            return self.get_item_arcface(index)
        elif self.mode == 'train-infonce':
            return self.get_item_infoNCE(index)
        elif self.mode == 'train-triplet':
            return self.get_item_triplet(index)
        elif self.mode == 'test':
            return self.get_item_test(index)
        else:
            raise Exception('No such mode!')

    def get_item_pam(self, index):
        if self.ml > self.mu or self.ml < 0:
            raise Exception('Margin of loss is not valid!')
        file = self.files[index]
        path = os.path.join(self.root_dir, file)
        data = np.load(path)
        label = int(file.split('-')[0]) - 1
        is_anchor = 0
        mid_start = max((len(data) - self.max_len) // 2, 0)
        if int(file.split('-')[1].split('.')[0]) == 0:
            start_index = mid_start
            chunk_len = min(len(data), self.max_len)
            overlap_len = chunk_len
            is_anchor = 1
        else:
            start_index, chunk_len, overlap_len = generate_chunk(len(data), self.max_len)
        margin = (self.mu - self.ml) * overlap_len / min(len(data), self.max_len) + self.ml
        sequence = np.zeros((self.max_len, self.feature_dim))
        sequence[:chunk_len] = data[start_index: start_index + chunk_len]
        padding_mask = np.zeros((self.max_len), dtype=np.bool_)
        padding_mask[chunk_len:] = 1

        return np.float32(sequence), np.int64(label), np.float32(margin), np.bool_(is_anchor), np.bool_(padding_mask)


    def get_item_infoNCE(self, index):
        file = self.files[index]
        org_path = os.path.join(self.root_dir, file)
        aug_num = random.randint(1, 9)
        aug_path = org_path[:-6] + f'{aug_num:02d}.npy'
        org_data = np.load(org_path)
        aug_data = np.load(aug_path)
        mid_start = max((len(aug_data) - self.max_len) // 2, 0)
        start_index, chunk_len, _ = generate_chunk(len(aug_data), self.max_len)
        org_seq, aug_seq = np.zeros((self.max_len, self.feature_dim)), np.zeros((self.max_len, self.feature_dim))
        org_seq[:min(len(org_data), self.max_len)] = org_data[mid_start: mid_start + min(len(org_data), self.max_len)]
        aug_seq[:chunk_len] = aug_data[start_index: start_index + chunk_len]
        org_mask, aug_mask = np.zeros((self.max_len), dtype=np.bool_), np.zeros((self.max_len), dtype=np.bool_)
        org_mask[min(len(org_data), self.max_len):] = 1
        aug_mask[chunk_len:] = 1

        return np.float32(org_seq), np.float32(aug_seq), np.bool_(org_mask), np.bool_(aug_mask)
    

    def get_item_triplet(self, index):
        file = self.files[index]
        a_index = int(file.split('-')[0])
        a_num = random.randint(1, 9)
        a_path = os.path.join(self.root_dir, f'{a_index}-{a_num:02d}.npy')
        p_path = os.path.join(self.root_dir, file)       
        n_index = a_index
        while n_index == a_index:
            n_index = random.randint(1, self.class_nums)
        n_path = os.path.join(self.root_dir, f'{n_index}-00.npy')
        
        a_data, p_data, n_data = np.load(a_path), np.load(p_path), np.load(n_path)
        a_mid_start = max((len(a_data) - self.max_len) // 2, 0)
        n_mid_start = max((len(n_data) - self.max_len) // 2, 0)
        a_start, a_len, _ = generate_chunk(len(a_data), self.max_len)

        a_seq, p_seq, n_seq = np.zeros((self.max_len, self.feature_dim)), np.zeros((self.max_len, self.feature_dim)), np.zeros((self.max_len, self.feature_dim))
        a_seq[:a_len] = a_data[a_start: a_start + a_len]
        p_seq[:min(len(p_data), self.max_len)] = p_data[a_mid_start: a_mid_start + min(len(p_data), self.max_len)]
        n_seq[:min(len(n_data), self.max_len)] = n_data[n_mid_start: n_mid_start + min(len(n_data), self.max_len)]
        
        a_mask, p_mask, n_mask = np.zeros((self.max_len), dtype=np.bool_), np.zeros((self.max_len), dtype=np.bool_), np.zeros((self.max_len), dtype=np.bool_)
        a_mask[a_len:] = 1
        p_mask[min(len(p_data), self.max_len):] = 1
        n_mask[min(len(n_data), self.max_len):] = 1

        return np.float32(a_seq), np.float32(p_seq), np.float32(n_seq), np.bool_(a_mask), np.bool_(p_mask), np.bool_(n_mask)


    def get_item_arcface(self, index):
        if self.ml > self.mu or self.ml < 0:
            raise Exception('Margin of loss is not valid!')
        file = self.files[index]
        path = os.path.join(self.root_dir, file)
        data = np.load(path)
        label = int(file.split('-')[0]) - 1

        mid_start = max((len(data) - self.max_len) // 2, 0)
        if int(file.split('-')[1].split('.')[0]) == 0:
            start_index = mid_start
            chunk_len = min(len(data), self.max_len)
            overlap_len = chunk_len
        else:
            start_index, chunk_len, overlap_len = generate_chunk(len(data), self.max_len)

        margin = (self.mu - self.ml) * overlap_len / min(len(data), self.max_len) + self.ml
        sequence = np.zeros((self.max_len, self.feature_dim), np.float32)
        sequence[:chunk_len] = data[start_index: start_index + chunk_len]
        padding_mask = np.zeros((self.max_len), dtype=np.bool_)   
        padding_mask[chunk_len:] = 1    

        return np.float32(sequence), np.int64(label), np.float32(margin), np.bool_(padding_mask)


    def get_item_test(self, index):
        file = self.files[index]
        path = os.path.join(self.root_dir, file)
        data = np.load(path)
        sequence = np.zeros((self.max_len, self.feature_dim), dtype=np.float32)
        length = min(len(data), self.seq_len)
        if self.seq_len == self.max_len:
            chunk_len = length
        else:
            chunk_len = self.seq_len
        sequence[:min(length, self.seq_len)] = data[self.start_id[index]: self.start_id[index] + min(length, self.seq_len)]
        padding_mask = np.zeros((self.max_len), dtype=np.bool_)
        padding_mask[chunk_len:] = 1

        return np.float32(sequence), np.bool_(padding_mask)


    def __len__(self):
        return len(self.files)
