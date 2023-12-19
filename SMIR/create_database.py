import os
import pickle
import shelve
import time
import pandas as pd
import torchaudio
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from models.mt import MT
from config import Config
from annoy import AnnoyIndex
from torchvision import transforms
import cv2
import torch.nn.functional as F
# from mydataset import SpectrogramFingerprintData
from torch.utils import data as Data
import faiss


config = Config.from_json_file("config.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MusicNetModel(config.emd_size, config.class_nums).cuda()


# model_mit = MIT(
#     input_dim=128,
#     output_dim=256,
#     dim=256,
#     depth=8,
#     heads=16,
#     dim_head=16,
#     mlp_dim=256,
#     dropout=0.1,
# ).cuda()
# model_mit = MICNN().cuda()


class SpectrogramFingerprintData(Data.Dataset):
    def __init__(self, paths, root_dir, max_len):
        self.paths = paths
        self.root_dir = root_dir
        self.max_len = max_len

    # def __getitem__(self, index):       
    #     file_path = self.paths[index]
    #     data = np.load(self.root_dir + "/" + file_path)
        # print(data.shape)
        # x = torch.tensor(data)
        # x = x.mean(dim = 0)
        # if len(data) == 0:
        #     x = torch.zeros(256)
        #     return np.float32(x.numpy())
        # x = torch.cat([x, x], dim = 0)
        # print(x.shape)
        # sum = np.zeros((1, 128))

        # for i in range(0, min(len(data), 100, self.max_len)):
        #     sum += data[i]

        # avg = sum/(min(len(data), 100, self.max_len))
        # new = np.concatenate((avg, avg), axis=1).squeeze(0)

        # sequence = np.zeros((100, 128))
        # for i in range(0, min(len(data), 100, self.max_len)):
        #     sequence[i] = data[i]

        # return np.float32(sequence)

    def __getitem__(self, index):       
        file_path = self.paths[index]
        data = np.load(self.root_dir + "/" + file_path)
        merge = np.zeros((len(data)-1, 256))
        center = np.zeros((1, 256))
        for j in range(min(len(data)-1, self.max_len)):
            merge[j] = np.concatenate((data[j], data[j+1]), axis=0)
        center[0] = np.mean(merge, axis=0)
        center = center.squeeze(0)
        # print(center.shape)

        return np.float32(center)

    def __len__(self):
        return len(self.paths)


def convert_audio_to_16k(input_file_path):
    waveform, sample_rate = torchaudio.load(input_file_path, normalize=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        torchaudio.save(input_file_path[:-4] + ".wav", waveform, 16000)
        # sound = AudioSegment.from_wav(input_file_path[:-4] + ".wav")
        # sound.export(input_file_path, format="mp3")
        # os.remove(input_file_path[:-4] + ".wav")


def split_audio_mp3(
    input_file_path, output_file_path, slice_duration=10000, interval=500
):
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    index = 1
    filename = os.path.basename(input_file_path)
    convert_audio_to_16k(input_file_path)
    audio = AudioSegment.from_file(input_file_path, format="mp3")
    length = (len(audio) // interval) * interval  # 舍弃结尾小于偏移间隔的片段
    slices = range(0, length - slice_duration + 1, interval)
    for j, start in enumerate(slices):
        end = slices[j] + slice_duration
        chunk = audio[start:end]
        chunk.export(
            os.path.join(output_file_path, filename[:-4] + f"-{index:02d}.mp3"),
            format="mp3",
        )
        index += 1


def split_audio_wav(
    input_file_path, output_file_path, slice_duration=10000, interval=500
):
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    index = 1
    filename = os.path.basename(input_file_path)
    convert_audio_to_16k(input_file_path)
    audio = AudioSegment.from_file(input_file_path, format="wav")
    length = (len(audio) // interval) * interval  # 舍弃结尾小于偏移间隔的片段
    slices = range(0, length - slice_duration + 1, interval)
    for j, start in enumerate(slices):
        end = slices[j] + slice_duration
        chunk = audio[start:end]
        chunk.export(
            os.path.join(output_file_path, filename[:-4] + f"-{index:02d}.wav"),
            format="wav",
        )
        index += 1


def generate_spectrogram(input_file_path, output_file_path):
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    for filename in os.listdir(input_file_path):
        file_path = input_file_path + "/" + filename

        waveform, sample_rate = torchaudio.load(file_path, normalize=True)
        transform = torchaudio.transforms.Spectrogram(n_fft=800)  # 创建一个Spectrogram对象
        spectrogram = transform(waveform)  # 对音频信号进行变换

        plt.imshow(spectrogram.log2()[0, :, :].numpy())
        plt.axis("off")
        output_path = output_file_path + "/" + filename[:-4] + ".jpg"
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.clf()
        plt.close()


def mp3_to_wav_mono(input_file_path, output_file_path):
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    for filename in os.listdir(input_file_path):
        new_filename = filename[:-3] + "wav"
        sound, sample_rate = torchaudio.load(input_file_path + "/" + filename)
        if sound.shape[0] == 2:
            sound_mono = torch.mean(sound, dim=0, keepdim=True)
            torchaudio.save(
                output_file_path + "/" + new_filename,
                sound_mono,
                sample_rate=sample_rate,
            )


def wav_to_wav_mono(input_file_path, output_file_path):
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    for filename in os.listdir(input_file_path):
        sound, sample_rate = torchaudio.load(input_file_path + "/" + filename)
        if len(sound.shape) == 2:
            sound_mono = torch.mean(sound, dim=0)
            torchaudio.save(
                output_file_path + "/" + filename,
                sound_mono.unsqueeze(0),
                sample_rate=sample_rate,
            )


def mp3_to_wav(input_file_path, output_file_path):
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    for filename in os.listdir(input_file_path):
        new_filename = filename[:-3] + "wav"
        sound = AudioSegment.from_mp3(input_file_path + "/" + filename)
        sound.export(output_file_path + "/" + new_filename, format="wav")


def get_tensor(input_file_path):
    pbar = tqdm.tqdm(os.listdir(input_file_path))
    pbar.set_description('Processing:')
    model.eval()

    for filename in pbar:

        wav_path = input_file_path + "/" + filename
        # convert_audio_to_16k(wav_path)
        waveform, sr = torchaudio.load(wav_path, normalize=True)
        length = (len(waveform[0]) // sr) * sr
        slices = range(0, length - 2 * sr + 1, sr)

        for j, start in enumerate(slices):
            end = slices[j] + 2 * sr
            chunk = waveform[0][start:end].unsqueeze(0)

            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=1024, win_length=1024, hop_length=256, n_mels=128, f_min=300, f_max=4000
            )(chunk)

            db_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            db_spec = db_spec.squeeze(0)
            mel_spec = (db_spec - db_spec.min()) / (db_spec.max() - db_spec.min()) * 255
            mel_spec[mel_spec < 0.2 * mel_spec.mean()] = 0
            mel_spec = mel_spec.numpy().astype(np.uint8)

            img = cv2.applyColorMap(mel_spec, cv2.COLORMAP_MAGMA)
            img = np.transpose(img, (2, 0, 1))
            spec = torch.from_numpy(np.float32(img)).unsqueeze(0)

            input_x = spec.to("cuda")
            song_tag = filename[:-4]+f"-{j+1:02d}"
            feature = model(input_x)
            feature = feature.cpu().detach().numpy()

            data_list = [song_tag]
            for i in range(128):
                data_list.append(feature[0][i])
            data = pd.DataFrame([data_list])
            data.to_csv("./database/small/chunk-tensor_v2.csv", mode="a", header=False, index=False)

        pbar.update()


# def get_tensor_mit(input_file_path, csv_path):
    # model_mit.eval()

    # # T1 = time.time()
    # # data_raw = load_data(input_file_path)
    # # T2 = time.time()
    # # print(f"load data time: {T2-T1}")
    # # with shelve.open("./database/chunk-tensor-db/chunk-tensor", "c") as db:
    # #     for key, value in data_raw.items():
    # #         db[key] = value
    # # T3 = time.time()
    # # print(f"save data time: {T3-T2}")

    # with shelve.open("./database/chunk-tensor-db/chunk-tensor", "r") as db:
    #     for key, value in tqdm.tqdm(db.items()):
    #         seq = np.zeros((50, 128))
    #         for i in range(len(value)):
    #             seq[i] = value[i]
    #             if i >= 49:
    #                 break

    #         seq = np.float32(seq)
    #         seq = torch.from_numpy(seq).unsqueeze(0)
    #         input_x = seq.to("cuda")

    #         song_tag = int(key)
    #         feature = model_mit(input_x)

    #         feature = F.normalize(feature.view(feature.shape[0],256))
    #         # feature = feature.view(feature.shape[0],256)
    #         feature = feature.cpu().detach().numpy()

    #         data_list = [song_tag]
    #         for i in range(256):
    #             data_list.append(feature[0][i])
    #         data = pd.DataFrame([data_list])
    #         data.to_csv(csv_path, mode="a", header=False, index=False)


def create_database():
    df = pd.read_csv("./database/small/chunk-tensor_v2.csv")
    f = 128
    t = AnnoyIndex(f, "euclidean")

    for row in tqdm.tqdm(df.iterrows()):
        song_tag = row[1][0]
        index = int(song_tag.split("-")[0].lstrip("0")) * 100 + int(
            song_tag.split("-")[1].lstrip("0")
        )
        vector = row[1][1:].tolist()
        t.add_item(index, vector)

    t.build(10)
    t.save("./database/small/database_v2.ann")


def create_database_mit(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path)
    f = 256
    t = AnnoyIndex(f, "angular")

    for row in df.iterrows():
        song_tag = row[1][0]
        index = int(song_tag)
        vector = row[1][1:].tolist()
        t.add_item(index, vector)
    
    t.build(10)
    t.save(output_file_path)


def get_vector_mit(output_root_dir, max_len = 100):
    # model_mit.eval()
    db_data = SpectrogramFingerprintData(paths=os.listdir(output_root_dir), root_dir=output_root_dir, max_len = max_len)
    db_data = Data.DataLoader(db_data, shuffle=False, batch_size=config.batch_size)
    db_nb = len(db_data)
    pbar = tqdm.tqdm(db_data, total=db_nb)

    arr_shape = (len(os.listdir(output_root_dir)), 256)
    arr = np.memmap(output_root_dir +".mm",
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    np.save(output_root_dir + "_shape.npy", arr_shape) 
    
    # for i, vector in enumerate(db_data):
        
    for i, vector in enumerate(pbar):
        emb = F.normalize(vector.cuda()).detach().cpu()
        arr[i * config.batch_size:(i + 1) * config.batch_size, :] = emb.numpy()

    print(f"=== Succesfully stored {arr_shape[0]} fingerprint to {os.path.dirname(output_root_dir)} ===")
    arr.flush(); del(arr)


def retrieval_new(query_name):
    T1 = time.time()
    # model_mit.eval()
    db_shape = np.load("./database/db_shape.npy")
    db = np.memmap("./database/db.mm", dtype='float32', mode='r+', shape=(db_shape[0], db_shape[1]))
    query_shape = np.load(f"./runs/retrieval/test/{query_name}_shape.npy")
    query = np.memmap(f"./runs/retrieval/test/{query_name}.mm", dtype='float32', mode='r+', shape=(query_shape[0], query_shape[1]))

    db_index = np.load("./database/db_inf.npy")
    db_index_map = dict()
    for i in range(len(db_index)):
        db_index_map[db_index[i][0]] = i

    xb_len = db_shape[0]
    xb = np.zeros((xb_len, 256))
    for i in range(xb_len):
            xb[i] = db[i]

    index = faiss.IndexFlatIP(256)
    index.add(xb)

    correct_num_1 = 0
    correct_num_10= 0
    filenames = os.listdir("./runs/retrieval/test/db")
    for ti in range(query_shape[0]):
        qi = int(filenames[ti][:-4])
        q = np.zeros((1, 256))
        q[0] = query[ti]
        D, I = index.search(q, 10)

        if db_index_map[qi] == I[0][0]:
            correct_num_1 += 1
            correct_num_10 += 1
        elif db_index_map[qi] in I[0]:
            correct_num_10 += 1
        # else:
        #     print(qi, db_index_map[qi], I)

    T2 = time.time()
    print("程序运行时间:%s毫秒" % ((T2 - T1) * 1000))
    print("top1 accuracy:", correct_num_1 / query_shape[0], "   ", "top10 accuracy:", correct_num_10 / query_shape[0])




if __name__ == "__main__":
    # input_path = "./database/song-wav"
    # output_path = "./database/chunk-wav"

    # mp3_to_wav_mono(input_path, output_path)

    # for filename in os.listdir(input_path):
    #     input_file_path = input_path + '/' + filename
    #     split_audio_wav(input_file_path, output_path, interval=1000)

    # generate_spectrogram("./database/chunk-mp3", "./database/chunk-image")
    # mp3_to_wav_mono("./database/chunk-mp3", "./database/chunk-wav")


    # Fingerprint
    # checkpoint = torch.load("./runs/train_v2/exp-9-5/musicNet.pth")
    # model.load_state_dict(checkpoint["model"], False)

    # get_tensor("./database/small/wav")

    # create_database()


    # # MIT
    # checkpoint = torch.load("./runs/mit/exp/mit.pth")
    # # checkpoint = torch.load("./runs/mit_chunk/exp/mit.pth")
    # model_mit.load_state_dict(checkpoint["model"], False)
    # get_vector_mit("./database/db")
    # get_vector_mit("./runs/retrieval/test/query")
    # get_vector_mit("./runs/retrieval/test/db-tr-aug")
    # get_vector_mit("./runs/retrieval/test/db-ts-aug")


    for i in range(1, 2):
        # checkpoint = torch.load(f"./runs/mit/exp/mit_{i}.pth")
        # model_mit.load_state_dict(checkpoint["model"])
        get_vector_mit("./database/db")
        get_vector_mit("./runs/retrieval/test/query", max_len=18)
        get_vector_mit("./runs/retrieval/test/db-tr-aug", max_len=18)
        get_vector_mit("./runs/retrieval/test/db-ts-aug", max_len=18)

        print(f"----------------------epoch: {i}----------------------")
        retrieval_new("db-tr-aug")
        retrieval_new("db-ts-aug")
        retrieval_new("query")


    # get_vector_mit("./runs/retrieval/test/db-tr-aug")
    # retrieval_new("db-tr-aug")
    # get_vector_mit("./database/db")