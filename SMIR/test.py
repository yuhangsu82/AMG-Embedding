# import csv
# import glob
# from pathlib import Path
# import shelve
# import shutil
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from tqdm import tqdm
# from model import MusicNetModel
# from torch.nn import functional as F
# # from train_mit import load_data
# from train import generate_triplet
# import torchaudio.functional as F
# from torch.utils import data as Data
# from config import Config
# from torch import nn
# from pydub import AudioSegment
# import torchaudio
# import os
# import random
# import ast
# import pandas as pd
# from torch_audiomentations import (
#     Compose,
#     Gain,
#     AddColoredNoise,
#     ApplyImpulseResponse,
#     AddBackgroundNoise,
#     HighPassFilter,
#     LowPassFilter,
# )
# import cv2
# from PIL import Image
# from torchvision import transforms
# import math
# from torchaudio.utils import download_asset
# # from create_database import split_audio_wav, mp3_to_wav_mono, wav_to_wav_mono
# import librosa
# import pickle
# from config import Config
# from tqdm import tqdm
# from train_mit import load_data

# def draw_fig(loss, acc, name, epoch):
#     epochs = range(1, epoch + 1)

#     # 绘制loss曲线
#     plt.cla()
#     plt.title(name + " loss vs. epoch", fontsize=20)
#     plt.xticks(np.arange(0, epoch + 1))
#     plt.plot(epochs, loss, ".-")
#     plt.xlabel("epoch", fontsize=20)
#     plt.ylabel(name + " loss", fontsize=20)
#     plt.grid()
#     plt.savefig("./runs/train/" + name + "_loss.png")

#     # 绘制acc曲线
#     plt.cla()
#     plt.title(name + " accuracy vs. epoch", fontsize=20)
#     plt.xticks(np.arange(0, epoch + 2))
#     plt.plot(epochs, acc, ".-")
#     plt.xlabel("epoch", fontsize=20)
#     plt.ylabel(name + " accuracy", fontsize=20)
#     plt.grid()
#     plt.savefig("./runs/train/" + name + " _accuracy.png")


# # 生成元组中正样本序号
# def generate_random_number_positive(index):
#     random_number = index + random.randint(1, 3)
#     return random_number


# # 生成元组中负样本序号
# def generate_random_number_negative(index, num):
#     while True:
#         random_number = random.randint(0, num - 1)
#         if random_number not in range(index, index + 21):
#             return random_number

#     # 生成三元组
#     # def generate_triplet(input_file_path):
#     paths = []
#     triplet_path = input_file_path[:-11] + "triplet.txt"
#     if os.path.isfile(triplet_path):
#         with open(triplet_path, "r") as f:
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     data = ast.literal_eval(line.split(": ")[1])
#                     paths.append(data)

#         return paths

#     else:
#         filenames = os.listdir(input_file_path)
#         num = len(filenames)
#         paths_dict = dict()

#         for filename in filenames:
#             index = int(filename.split("k")[1].replace(".jpg", ""))

#             if index > num - 4:
#                 continue

#             paths_dict[index] = [input_file_path + "/" + filename]
#             positive_filename = (
#                 input_file_path
#                 + "/"
#                 + filename.split("k")[0]
#                 + f"k{generate_random_number_positive(index)}.jpg"
#             )
#             paths_dict[index].append(positive_filename)
#             negative_filename = (
#                 input_file_path
#                 + "/"
#                 + filename.split("k")[0]
#                 + f"k{generate_random_number_negative(index, num)}.jpg"
#             )
#             paths_dict[index].append(negative_filename)

#         with open(triplet_path, "w") as f:
#             for key, value in paths_dict.items():
#                 f.write(f"{key}: {value}\n")

#         paths = [value for value in paths_dict.values()]
#         return paths


# # 构建自定义数据集
# class MusicData(Data.Dataset):
#     def __init__(self, paths, augment=False):
#         self.paths = paths
#         self.augment = augment

#     def __getitem__(self, index):
#         a_path, p_path, n_path = self.paths[index]
#         a_waveform, a_sr = torchaudio.load(a_path, normalize=True)
#         p_waveform, p_sr = torchaudio.load(p_path, normalize=True)
#         n_waveform, n_sr = torchaudio.load(n_path, normalize=True)

#         if self.augment:
#             a_waveform = apply_augmentation(
#                 a_waveform.unsqueeze(0), sample_rate=16000
#             ).squeeze(0)
#         #     p_waveform = apply_augmentation(p_waveform.unsqueeze(0), sample_rate=16000).squeeze(0)
#         #     n_waveform = apply_augmentation(n_waveform.unsqueeze(0), sample_rate=16000).squeeze(0)

#         a_mel_spec = torchaudio.transforms.MelSpectrogram(
#             sample_rate=a_sr, n_fft=1024, win_length=1024, hop_length=512, n_mels=128
#         )(a_waveform)
#         p_mel_spec = torchaudio.transforms.MelSpectrogram(
#             sample_rate=p_sr, n_fft=1024, win_length=1024, hop_length=512, n_mels=128
#         )(p_waveform)
#         n_mel_spec = torchaudio.transforms.MelSpectrogram(
#             sample_rate=n_sr, n_fft=1024, win_length=1024, hop_length=512, n_mels=128
#         )(n_waveform)

#         resize = transforms.Resize(
#             [313, 313], interpolation=transforms.InterpolationMode.BILINEAR
#         )
#         a_db_spec = resize(
#             torchaudio.transforms.AmplitudeToDB()(a_mel_spec).unsqueeze(0)
#         )
#         p_db_spec = resize(
#             torchaudio.transforms.AmplitudeToDB()(p_mel_spec).unsqueeze(0)
#         )
#         n_db_spec = resize(
#             torchaudio.transforms.AmplitudeToDB()(n_mel_spec).unsqueeze(0)
#         )

#         a_db_spec = a_db_spec[:, :, 50:288, :].squeeze(0).squeeze(0)
#         p_db_spec = p_db_spec[:, :, 50:288, :].squeeze(0).squeeze(0)
#         n_db_spec = n_db_spec[:, :, 50:288, :].squeeze(0).squeeze(0)

#         a_mel_spec = (
#             (a_db_spec - a_db_spec.min()) / (a_db_spec.max() - a_db_spec.min()) * 255
#         )
#         p_mel_spec = (
#             (p_db_spec - p_db_spec.min()) / (p_db_spec.max() - p_db_spec.min()) * 255
#         )
#         n_mel_spec = (
#             (n_db_spec - n_db_spec.min()) / (n_db_spec.max() - n_db_spec.min()) * 255
#         )

#         # a_mel_spec[a_mel_spec < 0.6 * a_mel_spec.mean()] = 0
#         # p_mel_spec[p_mel_spec < 0.6 * a_mel_spec.mean()] = 0
#         # n_mel_spec[n_mel_spec < 0.6 * a_mel_spec.mean()] = 0

#         a_mel_spec = a_mel_spec.numpy().astype(np.uint8)
#         p_mel_spec = p_mel_spec.numpy().astype(np.uint8)
#         n_mel_spec = n_mel_spec.numpy().astype(np.uint8)

#         a_img = cv2.applyColorMap(a_mel_spec, cv2.COLORMAP_MAGMA)
#         p_img = cv2.applyColorMap(p_mel_spec, cv2.COLORMAP_MAGMA)
#         n_img = cv2.applyColorMap(n_mel_spec, cv2.COLORMAP_MAGMA)

#         a_img = np.transpose(a_img, (2, 0, 1))
#         p_img = np.transpose(p_img, (2, 0, 1))
#         n_img = np.transpose(n_img, (2, 0, 1))

#         # a_mel_spec = torch.where(a_db_spec == -np.inf, torch.zeros_like(a_db_spec), a_db_spec)
#         # p_mel_spec = torch.where(p_db_spec == -np.inf, torch.zeros_like(p_db_spec), p_db_spec)
#         # n_mel_spec = torch.where(n_db_spec == -np.inf, torch.zeros_like(n_db_spec), n_db_spec)

#         # a_mel_spec = ((a_db_spec - a_db_spec.min()) / (a_db_spec.max() - a_db_spec.min()))
#         # p_mel_spec = ((p_db_spec - p_db_spec.min()) / (p_db_spec.max() - p_db_spec.min()))
#         # n_mel_spec = ((n_db_spec - n_db_spec.min()) / (n_db_spec.max() - n_db_spec.min()))

#         # a_mel_spec = torch.stack([a_db_spec] * 3, dim=0)
#         # p_mel_spec = torch.stack([p_db_spec] * 3, dim=0)
#         # n_mel_spec = torch.stack([n_db_spec] * 3, dim=0)

#         return np.float32(a_img), np.float32(p_img), np.float32(n_img)

#     def __len__(self):
#         return len(self.paths)


# def get_threshold(threshold):
#     config = Config.from_json_file("config.json")
#     model = MusicNetModel(config.emd_size, config.class_nums).cuda()
#
#     test_file_path = "./datasets/test/triplet.txt"
#     test_paths = generate_triplet(test_file_path)
#     test_data = MusicData(test_paths)
#     test_data = Data.DataLoader(test_data, shuffle=True, batch_size=config.batch_size)
#
#     checkpoint = torch.load('./runs/train/exp1/musicnet.pth')
#     model.load_state_dict(checkpoint['model'])
#     model.eval()
#     val_nb = len(test_data)
#     pbar_val = tqdm(test_data, total=val_nb)
#     ta = fa = 0
#     total_num_val = len(test_data.dataset)
#
#     with torch.no_grad():
#         for step, (a_x, p_x, n_x) in enumerate(pbar_val):
#             a_x = a_x.cuda()
#             p_x = p_x.cuda()
#             n_x = n_x.cuda()
#
#             transform = torchaudio.transforms.Spectrogram(n_fft=800).cuda()
#
#             a_spec = normalize(transform(a_x).log2())
#             p_spec = normalize(transform(p_x).log2())
#             n_spec = normalize(transform(n_x).log2())
#
#             print(a_spec)
#             print(p_spec)
#             print(n_spec)
#
#             a_out, p_out, n_out = model(a_spec), model(p_spec), model(n_spec)
#
#             print(a_out)
#             print(p_out)
#             print(n_out)
#
#             s_d = F.pairwise_distance(a_out, p_out)
#             n_d = F.pairwise_distance(a_out, n_out)
#             print(s_d)
#             print(n_d)
#
#             thing1 = (s_d <= threshold).flatten().cpu()
#             thing2 = (n_d <= threshold).flatten().cpu()
#
#             ta += torch.sum(thing1).item()
#             fa += torch.sum(thing2).item()
#
#     VAL = ta / total_num_val
#     FAR = fa / total_num_val
#     print("T: ", threshold, " ---- VAL: ", VAL, " ------ FAR: ", FAR)
#
#     with open("./runs/train/exp1/VAL-FAR.txt", 'a') as f:
#         f.write(f"T: {threshold}  ---- VAL: {VAL}  ------ FAR: {FAR}\n")


# for i in range(16):
#     threshold = 3 + i * 0.5
#     get_threshold(threshold)

# def generate_random_number_positive(i):
#     random_number = i + random.randint(0, 3)
#     return random_number
#
#
# i = 0
# j = 77

# generate_triplet("./datasets/test/image")
#
#
# paths = []
# triplet = dict()
# with open("./datasets/test/triplet.txt", 'r') as f:
#     for line in f:
#         line = line.strip()
#         if line:
#             data = ast.literal_eval(line.split(': ')[1])
#             triplet[int(line.split(': ')[0])] = data
#             paths.append(data)

# for i in range(1, 100):
#
#     a = b = c = 4000
#     while a > j:
#         a = generate_random_number_positive(j-2)
#     while b > j:
#         b = generate_random_number_positive(j-1)
#     while c > j:
#         c = generate_random_number_positive(j)
#     triplet[j-2][1] = "./datasets/val/image/val_chunk"+str(a)+".jpg"
#     triplet[j - 1][1] = "./datasets/val/image/val_chunk" + str(b) + ".jpg"
#     triplet[j ][1] = "./datasets/val/image/val_chunk" + str(c) + ".jpg"
#     print(triplet[j-2])
#     print(triplet[j-1])
#     print(triplet[j])
#
#     with open("./datasets/val/triplet1.txt", "w") as f:
#         for key, value in triplet.items():
#             f.write(f"{key}: {value}\n")
#
#     j += 78
#     if j > 974:
#         break


# list_a = [39, 80, 120, 160, 200, 241, 282, 322, 363, 403, 443, 484, 524, 564, 604, 645, 686, 726, 766, 806, 847, 888,
#           929, 970]
#
# for item in list_a:
#     del triplet[item]
#     triplet[item-1][1] = "./datasets/test/image/test_chunk" + str(item) + ".jpg"
#     triplet[item-2][1] = "./datasets/test/image/test_chunk" + str(item-2+random.randint(1, 2)) + ".jpg"
#
#
# with open("./datasets/test/triplet1.txt", "w") as f:
#         for key, value in triplet.items():
#             f.write(f"{key}: {value}\n")


# waveform, sample_rate = torchaudio.load("./runs/retrieval/test_1_volume=75.mp3", normalize=True)
# resampler = torchaudio.transforms.Resample(sample_rate, 16000)
# waveform = resampler(waveform)
#
# torchaudio.save("./runs/retrieval/test_1_volume=75.wav", waveform, 16000)
# print(sample_rate)
#
# sound = AudioSegment.from_file("./database/song/0005.mp3", format="mp3")
# chunk = sound[0:19000]
# chunk.export("./runs/retrieval/split_5.mp3", format="mp3")

# waveform, sample_rate = torchaudio.load("./runs/retrieval/test_1_volume=75.mp3", normalize=True)
# resampler = torchaudio.transforms.Resample(sample_rate, 16000)
# waveform = resampler(waveform)
#
# data = waveform.numpy().T.copy(order = 'C')
#
# sound = AudioSegment(data, sample_width=2, frame_rate=16000, channels=1)
# sound.export("./runs/retrieval/test_1_volume=751.mp3", format="mp3")
# print(sample_rate)


# path = "./database/chunk-image"
# print(os.path.dirname(path))


# audio = AudioSegment.from_file("./runs/retrieval/test/test_5_volume=80.mp3", format="mp3")
# audio = audio + 6
# volume = audio.rms
# print("音量值：", volume)
# audio.export("./runs/retrieval/test/test_5_volume=80+6.mp3", format="mp3")

# for filename in os.listdir("./database/song"):
#     file_path = "./database/song/" + filename
#     audio = AudioSegment.from_file(file_path, format="mp3")
#     volume = audio.rms
#     print(f"{filename}-音量值：", volume)

# waveform, sample_rate = torchaudio.load("./runs/retrieval/alternative/test_1_volume=50.mp3",
#                                         normalize=True)
# transform = torchaudio.transforms.Spectrogram(n_fft=800)  # 创建一个Spectrogram对象
# spectrogram = transform(waveform)  # 对音频信号进行变换
# print(spectrogram.shape)
# print(type(spectrogram))
# print(spectrogram[0].numpy())
# print("--------------------------")
# print(spectrogram[1])
# print("--------------------------")
# print(spectrogram.log2()[0, :, :].numpy().size)
# print(spectrogram.log2()[0, :, :].numpy())
# plt.imshow(torch.randn(401, 401))
# plt.axis("off")
# plt.savefig("./runs/aaa.jpg", bbox_inches='tight', pad_inches=0)

# k= torch.load("./datasets/val/spectrogram/val_chunk1.pt")
# print(k)
# print(k.shape)


# for filename in os.listdir("./datasets/val/mp3"):
#     file = "./datasets/val/mp3/" + filename
#     audio = AudioSegment.from_file(file, format="mp3")
#     if audio.channels != 1:
#         print(audio.channels)


# apply_augmentation = Compose(
#     transforms=[
#         Gain(
#             min_gain_in_db=24.0,
#             max_gain_in_db=25.0,
#             p=1,
#         ),
#         AddColoredNoise(min_snr_in_db=3.0,max_snr_in_db=10.0, min_f_decay=0, p=1)
#     ]
# )
# #
# waveform, sample_rate = torchaudio.load("./database/chunk-wav/0001-01.wav", normalize=True)
# # print(waveform.shape)
#
# # ./runs/retrieval/alternative/split.mp3
# # print(sample_rate)
#
# samples = apply_augmentation(waveform.unsqueeze(0), sample_rate=16000)
# torchaudio.save("./database/chunk-wav/0001-01r.wav", samples.squeeze(0), 16000)

# waveform, sample_rate = torchaudio.load("./runs/retrieval/alternative/test1-100.mp3", normalize=True)
#
# waveform_1 = waveform[0:]
# waveform_2 = waveform[1:]
#
# torchaudio.save("./runs/retrieval/alternative/test1-100-1.wav", waveform_1, 48000)
# torchaudio.save("./runs/retrieval/alternative/test1-100-2.wav", waveform_2, 48000)


# file_path = "./datasets/train/wav/train_chunk1.wav"
#
# waveform, sample_rate = torchaudio.load(file_path, normalize=True)
# transform = torchaudio.transforms.Spectrogram(n_fft=800)  # 创建一个Spectrogram对象
# spectrogram = transform(waveform)  # 对音频信号进行变换
#
# print(spectrogram.log2()[0, :, :].shape)
# print(spectrogram.log2()[0, :, :].numpy())
#
# plt.figure(dpi=100)
# plt.imshow(spectrogram.log2()[0, :, :].numpy())
# plt.axis("off")
# output_path = "./train_chunk1.jpg"
# # plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
#
# plt.savefig(output_path, dpi=100)
# # mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'))
# # plt.show()
# plt.clf()
# plt.close()
#
# # 读取图像
# img = Image.open(output_path)
# # 转换为numpy数组
# arr = np.array(img)
# # 初始化裁剪后的数组
# cropped = arr.copy()
# # 从上到下遍历每一行，如果该行全是零值，则删除该行
# while np.all(cropped[0,:,:] == 255):
#     cropped = np.delete(cropped, 0, axis=0)
# # 从下到上遍历每一行，如果该行全是零值，则删除该行
# while np.all(cropped[-1,:,:] == 255):
#     cropped = np.delete(cropped, -1, axis=0)
# # 从左到右遍历每一列，如果该列全是零值，则删除该列
# while np.all(cropped[:,0,:] == 255):
#     cropped = np.delete(cropped, 0, axis=1)
# # 从右到左遍历每一列，如果该列全是零值，则删除该列
# while np.all(cropped[:,-1,:] == 255):
#     cropped = np.delete(cropped, -1, axis=1)
# # 转换回图像
# img = Image.fromarray(cropped)
# # 保存图像
# img.save(output_path)
#
# image = cv2.imread(output_path)
# image = np.transpose(image, (2, 0, 1))
# print(image.shape)
# print(image)

# # 创建一个401*401的tensor
# t = torch.rand(500, 500)
# plt.figure(dpi=100)
# plt.imshow(t)
# plt.axis("off")
# plt.savefig('image.png', dpi=100, bbox_inches='tight', pad_inches=0)


# 获取默认的颜色映射表
# cmap = plt.get_cmap()
# # 将颜色映射表转换为一个数据框
# df = pd.DataFrame(cmap.colors, columns=['Red', 'Green', 'Blue'])
# # 将数据框保存为一个csv文件
# df.to_csv('./cmap.csv', index=False)

# 假设你的tensor叫做x
# x = torch.randn(1, 1, 3, 4)  # 随机生成一个示例tensor

# print(x)
# # 计算x的最大值和最小值
#
# print(x[0].max())
# print(x[0].min())
# print(x[0][0][0][0])
# print(2 * (x[0][0][0][0] - x[0].min()) / (x[0].max() - x[0].min()) - 1)
# # x_new = nn.BatchNorm2d(1)(x)
# # bn = nn.BatchNorm2d(1, requires_grad=False)
# x_new = F.normalize(x, dim=3, p=float('inf'))
# # x_new = bn(x)
# print(x_new)
# print(x_new.shape)

# # 假设你的tensor叫做x
# x = torch.randn(2, 1, 401, 401) # 随机生成一个示例tensor
# print(x)


# def normalize_1(x):
#     # 使用normalize函数对第一个维度进行归一化，使用无穷范数（即最大值）
#     x_new = F.normalize(x, dim=0, p=float('inf'))
#     # 返回归一化后的tensor
#     return x_new
#
#
# # x1 = normalize(x)
# # x2 = normalize_1(x)
# # print(x1)
# # print(x2)
#
#
# def norm(x):
#     x_max = x.max()
#     x_min = x.min()
#     # 使用线性缩放的公式将x1归一化到[-1, 1]
#     x_new = 2 * (x - x_min) / (x_max - x_min) - 1
#     return x_new


# df = pd.read_csv("./cmap.csv")

# file_path = "./datasets/train/wav/train_chunk1.wav"

# waveform, sample_rate = torchaudio.load(file_path, normalize=True)
# transform = torchaudio.transforms.Spectrogram(n_fft=800)  # 创建一个Spectrogram对象
# spectrogram = transform(waveform).squeeze(0).log2()  # 对音频信号进行变换
# spectrogram_max = spectrogram.max()
# spectrogram_min = spectrogram.min()
# spectrogram = (spectrogram - spectrogram_min) / (spectrogram_max - spectrogram_min)


# def gray_to_color(spec):
#     width = len(spec[0])
#     height = len(spec)
#     spec_img = torch.zeros(3, width, height)
#     print("33333333333333333333333")

#     for i in range(0, width):
#         for j in range(0, height):
#             for ri in range(0, 256):
#                 if spec[i][j] < df.Red[ri]:
#                     spec_img[2][i][j] = ri
#                     break
#                 ri += 1

#             for gi in range(0, 256):
#                 if spec[i][j] < df.Green[gi]:
#                     spec_img[1][i][j] = gi
#                     break
#                 gi += 1

#             for bi in range(0, 256):
#                 if spec[i][j] <= df.Blue[bi]:
#                     spec_img[1][i][j] = bi
#                     break
#                 bi += 1
#             j += 1

#         i += 1
#         print(i)

#     print("12121212")
#     img = spec_img.numpy().transpose(1, 2, 0)
#     print(img)
#     cv2.imwrite("./test_spec.jpg", img)
#     print("11111111111111111111111111111111")


# gray_to_color(spectrogram)


# # 导入需要的库
# import cv2
# import torchaudio
# import numpy as np
# import matplotlib.pyplot as plt

# # 读取wav音频文件，返回波形数据和采样率
# waveform, sample_rate = torchaudio.load('path/to/wav/file.wav')

# # 计算短时傅里叶变换，返回复数的频谱数据
# spec = torchaudio.transforms.Spectrogram(n_fft=1024, power=None)(waveform)

# # 将复数的频谱数据转换为幅度和相位的形式
# spec_amp = torchaudio.functional.complex_norm(spec) # 幅度
# spec_phase = torchaudio.functional.angle(spec) # 相位

# # 将幅度和相位的数据转换为numpy数组，并取对数
# spec_amp_np = np.log(spec_amp.numpy()[0]) # 取第一个通道的数据
# spec_phase_np = np.log(spec_phase.numpy()[0])

# # 使用opencv的applyColorMap函数，将幅度和相位的数据映射到彩色图像上
# spec_amp_img = cv2.applyColorMap(np.uint8(255 * spec_amp_np / spec_amp_np.max()), cv2.COLORMAP_JET)
# spec_phase_img = cv2.applyColorMap(np.uint8(255 * spec_phase_np / spec_phase_np.max()), cv2.COLORMAP_JET)

# # 显示或保存彩色图像
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(spec_amp_img)
# plt.title('Spectrogram amplitude')
# plt.subplot(1, 2, 2)
# plt.imshow(spec_phase_img)
# plt.title('Spectrogram phase')
# plt.show()
# cv2.imwrite('spec_amp_img.jpg', spec_amp_img)
# cv2.imwrite('spec_phase_img.jpg', spec_phase_img)


# waveform1, sr1 = torchaudio.load("./database/chunk-wav/0001-01.wav", normalize=True)

# # waveform2, sr2 = torchaudio.load("./runs/retrieval/0005-wav/0005-01.wav", normalize=True)

# # waveform3, sr3 = torchaudio.load("./datasets/train/wav/train_chunk1.wav", normalize=True)

# print(waveform1.shape)
# print(sr1)

# print(waveform2.shape)
# print(sr2)

# print(waveform3.shape)
# print(sr3)

# waveform, sr = torchaudio.load("./datasets/train/wav/train_chunk1.wav")
# print(sr)

# mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, win_length=1024,
#                                                        hop_length=512, n_mels=128)(waveform)

# print(mel_spectrogram.shape)

# # resize = transforms.Resize([313, 313], interpolation=transforms.InterpolationMode.BILINEAR)
# db_spectrogram = resize(torchaudio.transforms.AmplitudeToDB()(mel_spectrogram).unsqueeze(0))

# db_spectrogram = db_spectrogram.squeeze(0).squeeze(0)

# mel_spec = db_spectrogram
# spectrogram = torch.where(mel_spec == -np.inf, torch.zeros_like(mel_spec), mel_spec)
# spectrogram = ((spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min()) * 255)


# apply_augmentation = Compose(
#     transforms=[
#         Gain(
#             min_gain_in_db=-25.0,
#             max_gain_in_db=25.0,
#             p=1,
#         ),
#         AddColoredNoise(p=1),
#         ApplyImpulseResponse
#         # AddColoredNoise(min_f_decay=1.0, p=0.5)
#         # # AddColoredNoise(min_f_decay=2, p=0.5)
#         # # AddColoredNoise(min_f_decay=-1, p=0.5)
#         # # AddColoredNoise(min_f_decay=-2, p=0.5)
#     ]
# )


# for filename in os.listdir("./runs/retrieval/example-mp3"):
#     filepath = "./runs/retrieval/example-mp3/"+filename
#     out_path = "./runs/retrieval/aaa/"+filename[:-3] + "wav"
#     waveform, sample_rate = torchaudio.load(filepath, normalize=True)
#     print(sample_rate)
#     # print(waveform.shape)

#     # ./runs/retrieval/alternative/split.mp3
#     # print(sample_rate)

#     samples = apply_augmentation(waveform.unsqueeze(0), sample_rate=16000)
#     torchaudio.save(out_path, samples.squeeze(0), 16000)


# # # 文件路径
# # path = "E:/new-dataset/new-data"

# # # 获取文件夹中的所有音频文件
# # audio_files = os.listdir(path)
# # audio_files = [audio_file for audio_file in audio_files if audio_file.endswith(".wav")]
# # print(audio_files)

# # # 拼接音频文件
# # mp3 = None
# # for audio_file in audio_files:
# #     # 读取音频文件
# #     audio = AudioSegment.from_wav(path + "/" + audio_file)

# #     # 拼接音频文件
# #     if mp3 is None:
# #         mp3 = audio
# #     else:
# #         mp3 += audio

# # # 导出音频文件
# # mp3.export("E:/output.wav", format="wav")

# waveform, sample_rate = torchaudio.load(filepath, normalize=True)
# print(sample_rate)
# # print(waveform.shape)

# # ./runs/retrieval/alternative/split.mp3
# # print(sample_rate)

# samples = apply_augmentation(waveform.unsqueeze(0), sample_rate=16000)
# rir_applied = F.fftconvolve(speech, rir)
# torchaudio.save(out_path, samples.squeeze(0), 16000)


# def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None):
#     waveform = waveform.numpy()

#     num_channels, num_frames = waveform.shape
#     time_axis = torch.arange(0, num_frames) / sample_rate

#     figure, axes = plt.subplots(num_channels, 1)
#     if num_channels == 1:
#         axes = [axes]
#     for c in range(num_channels):
#         axes[c].plot(time_axis, waveform[c], linewidth=1)
#         axes[c].grid(True)
#         if num_channels > 1:
#             axes[c].set_ylabel(f"Channel {c+1}")
#         if xlim:
#             axes[c].set_xlim(xlim)
#     figure.suptitle(title)
#     plt.show(block=False)


# def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
#     waveform = waveform.numpy()

#     num_channels, _ = waveform.shape

#     figure, axes = plt.subplots(num_channels, 1)
#     if num_channels == 1:
#         axes = [axes]
#     for c in range(num_channels):
#         axes[c].specgram(waveform[c], Fs=sample_rate)
#         if num_channels > 1:
#             axes[c].set_ylabel(f"Channel {c+1}")
#         if xlim:
#             axes[c].set_xlim(xlim)
#     figure.suptitle(title)
#     plt.show(block=False)


# SAMPLE_RIR = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
# rir_raw_1, sample_rate = torchaudio.load(SAMPLE_RIR)
# print(sample_rate)


# rir_raw, sample_rate = torchaudio.load("E:/code/SMIR/datasets/augmentation/rir_raw_3.wav")
# resampler = torchaudio.transforms.Resample(sample_rate, 16000)
# rir_raw = resampler(rir_raw)

# print(sample_rate)
# # plot_waveform(rir_raw, 16000, title="Room Impulse Response (raw)")
# # plot_specgram(rir_raw, 16000, title="Room Impulse Response (raw)")
# Audio(rir_raw, rate=16000)


# rir = rir_raw[:, int(16000 * 1.01): int(16000 * 1.3)]
# rir = rir / torch.norm(rir, p=2)

# # plot_waveform(rir, sample_rate, title="Room Impulse Response")
# print(rir.shape)
# print(rir)

# rir = torch.mean(rir, dim=0).unsqueeze(0)
# torchaudio.save("E:/code/SMIR/test/rir_3.wav",  rir, 16000)
# print(rir.shape)
# print(rir)

# speech, _ = torchaudio.load("E:/code/SMIR/test/0001-01.wav")
# augmented = F.fftconvolve(speech, rir)

# torchaudio.save("E:/code/SMIR/test/0001-01-rir-3.wav", augmented, 16000)


# ir_paths = ["./datasets/augmentation/ImpulseResponse/rir_1.wav",
#             "./datasets/augmentation/ImpulseResponse/rir_2.wav"]

# bk_paths = ["./test/bk-2.wav"]

# apply_augmentation = Compose(
#     transforms=[
#         # Gain(
#         #     min_gain_in_db=-15.0,
#         #     max_gain_in_db=25.0,
#         #     p=0.75,
#         # ),
#         # AddColoredNoise(p=1.0),
#         # ApplyImpulseResponse(ir_paths, p=0.75),
#         AddBackgroundNoise(bk_paths, p=0.75)

#     ]
# )

# for filename in os.listdir("./test/aaaa"):
#     filepath = "./test/aaaa/"+filename
#     out_path = "./test/bbbb/"+filename
#     waveform, sample_rate = torchaudio.load(filepath, normalize=True)
#     print(sample_rate)
#     # print(waveform.shape)

#     # ./runs/retrieval/alternative/split.mp3
#     # print(sample_rate)

#     samples = apply_augmentation(waveform.unsqueeze(0), sample_rate=16000)
#     torchaudio.save(out_path, samples.squeeze(0), 16000)


# sound, sample_rate = torchaudio.load("./test/bk-1.wav")
# sound_mono = torch.mean(sound, dim=0, keepdim=True)
# resampler = torchaudio.transforms.Resample(sample_rate, 16000)
# sound_mono = resampler(sound_mono)
# torchaudio.save("./test/bk-2.wav", sound_mono, sample_rate=16000)


# config = Config.from_json_file("config.json")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MusicNetModel(config.emd_size, config.class_nums).cuda()
# checkpoint = torch.load("./runs/train/exp/musicnet.pth")
# model.load_state_dict(checkpoint['model'])


# waveform_a, sr_a = torchaudio.load("./runs/retrieval/aaaaa/0001-04.wav", normalize=True)
# waveform_p, sr_p = torchaudio.load("./runs/retrieval/aaaaa/test1-95-mono-03.wav", normalize=True)

# mel_spec_a = torchaudio.transforms.MelSpectrogram(sample_rate=sr_a, n_fft=1024, win_length=1024,
#                                                   hop_length=512, n_mels=128)(waveform_a).cuda()

# mel_spec_p = torchaudio.transforms.MelSpectrogram(sample_rate=sr_p, n_fft=1024, win_length=1024,
#                                                   hop_length=512, n_mels=128)(waveform_p).cuda()

# resize = transforms.Resize([313, 313], interpolation=transforms.InterpolationMode.BILINEAR)
# db_spec_a = resize(torchaudio.transforms.AmplitudeToDB()(
#     mel_spec_a).unsqueeze(0)).squeeze(0).squeeze(0)
# db_spec_p = resize(torchaudio.transforms.AmplitudeToDB()(
#     mel_spec_p).unsqueeze(0)).squeeze(0).squeeze(0)

# # mel_spec = torch.where(db_spec == -np.inf, torch.zeros_like(db_spec), db_spec)

# mel_spec_a = ((db_spec_a - db_spec_a.min()) /
#               (db_spec_a.max() - db_spec_a.min()))

# mel_spec_p = ((db_spec_p - db_spec_p.min()) /
#               (db_spec_p.max() - db_spec_p.min()))

# a_x = torch.stack([mel_spec_a] * 3, dim=0).unsqueeze(0).cuda()

# p_x = torch.stack([mel_spec_p] * 3, dim=0).unsqueeze(0).cuda()


# a_out, p_out = model(a_x), model(p_x)

# dis = F.pairwise_distance(a_out, p_out)

# print(dis)


# split = [0]
# sum = 0

# i = 0
# for filename in os.listdir("E:/new-dataset/new-data"):
#     filepath = "E:/new-dataset/new-data/" + filename

#     waveform, sr = torchaudio.load(filepath, normalize=True)
#     # print(len(waveform[0]))
#     # print(sr)
#     sum += len(waveform[0])
#     split.append(sum-1)

# print(split)


# waveform, sr = torchaudio.load("E:/new.wav")
# resampler = torchaudio.transforms.Resample(48000, 16000)
# waveform = resampler(waveform)
# print(sr)
# print(waveform.shape)

# for i in range(len(split) - 1):
#     # 获取切割点的起始和结束位置
#     start = split[i]
#     end = split[i + 1]
#     # 按照第二个维度切割tensor，得到一个2*(end-start)的子tensor
#     slice = waveform[:, start:end]
#     filepath = f"E:/split-wav/song_{i+1}.wav"

#     torchaudio.save(filepath, slice, 16000)


# 生成元组中正样本序号

# def generate_random_number_positive(i):
#     random_number = i + random.randint(1, 3)
#     return random_number


# # 生成元组中负样本序号
# def generate_random_number_negative(i, num):
#     while True:
#         random_number = random.randint(1, num - 1)
#         if random_number not in range(i, i + 21):
#             return random_number


# # 生成三元组
# def generate_triplet(input_file_path):
#     paths = []
#     triplet_path = input_file_path[:-3] + "triplet.txt"
#     if os.path.isfile(triplet_path):
#         with open(triplet_path, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     data = ast.literal_eval(line.split(': ')[1])
#                     paths.append(data)

#         return paths

#     else:
#         filenames = os.listdir(input_file_path)
#         num = len(filenames)
#         paths_dict = dict()
#         with open(os.path.dirname(input_file_path) + "/chunk_inf.csv", 'r') as csvfile:
#             reader = csv.reader(csvfile)
#             next(reader)
#             barriers = set([int(row[1].split("-")[1]) for row in reader])

#         for index in range(1, num+1):
#             filename = "val_raw_chunk" + str(index) + ".wav"
#             if index > num - 4:
#                 continue

#             fname = "./datasets/val/wav_record/val_record_chunk"

#             paths_dict[index] = [input_file_path + "/" + filename]
#             positive_filename = fname + f"{generate_random_number_positive(index)}.wav"
#             paths_dict[index].append(positive_filename)
#             negative_filename = input_file_path + "/" + filename.split('k')[
#                 0] + f"k{generate_random_number_negative(index, num+1)}.wav"
#             paths_dict[index].append(negative_filename)

#             if index in barriers:
#                 a = b = num
#                 while a > index:
#                     a = generate_random_number_positive(index - 2)
#                 while b > index:
#                     b = generate_random_number_positive(index - 1)
#                 paths_dict[index - 2][1] = input_file_path + "/" + filename.split('k')[0] + f"k{a}.wav"
#                 paths_dict[index - 1][1] = input_file_path + "/" + filename.split('k')[0] + f"k{b}.wav"
#                 paths_dict[index][1] = input_file_path + "/" + filename.split('k')[0] + f"k{index}.wav"

#         with open(triplet_path, "w") as f:
#             for key, value in paths_dict.items():
#                 f.write(f"{key}: {value}\n")

#         paths = [value for value in paths_dict.values()]
#         return paths


# # 生成三元组
# def generate_triplet(input_file_path):
#     paths = []
#     triplet_path = input_file_path[:-3] + "triplet.txt"
#     if os.path.isfile(triplet_path):
#         with open(triplet_path, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     data = ast.literal_eval(line.split(': ')[1])
#                     paths.append(data)

#         return paths

#     else:
#         filenames = os.listdir(input_file_path)
#         num = len(filenames)
#         paths_dict = dict()
#         with open(os.path.dirname(input_file_path) + "/chunk_inf.csv", 'r') as csvfile:
#             reader = csv.reader(csvfile)
#             next(reader)
#             barriers = set([int(row[1].split("-")[1]) for row in reader])

#         for index in range(1, num+1):
#             filename = os.path.basename(os.path.dirname(input_file_path)) + "_chunk" + str(index) + ".wav"
#             if index > num - 4:
#                 continue

#             paths_dict[index] = [input_file_path + "/" + filename]
#             positive_filename = input_file_path + "/" + filename.split('k')[
#                 0] + f"k{generate_random_number_positive(index)}.wav"
#             paths_dict[index].append(positive_filename)
#             negative_filename = input_file_path + "/" + filename.split('k')[
#                 0] + f"k{generate_random_number_negative(index, num)}.wav"
#             paths_dict[index].append(negative_filename)

#             if index in barriers:
#                 a = b = num
#                 while a > index:
#                     a = generate_random_number_positive(index - 2)
#                 while b > index:
#                     b = generate_random_number_positive(index - 1)
#                 paths_dict[index - 2][1] = input_file_path + "/" + filename.split('k')[0] + f"k{a}.wav"
#                 paths_dict[index - 1][1] = input_file_path + "/" + filename.split('k')[0] + f"k{b}.wav"
#                 paths_dict[index][1] = input_file_path + "/" + filename.split('k')[0] + f"k{index}.wav"

#         with open(triplet_path, "w") as f:
#             for key, value in paths_dict.items():
#                 f.write(f"{key}: {value}\n")

#         paths = [value for value in paths_dict.values()]
#         return paths


# train_file_path = "./datasets/train/wav"
# train_paths = generate_triplet(train_file_path)


# def draw_mel_hui():
#     for filename in os.listdir("./test/0001-wav-aug"):
#         if filename != "csv" and filename != "photo":
#             file_path_audio = "./test/0001-wav-aug/" + filename
#             mel_path = "./test/photo/" + filename[:-4] + "-aug.png"
#             waveform, sr = torchaudio.load(file_path_audio)

#             mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, win_length=1024,
#                                                                    hop_length=512, n_mels=128)(waveform)

#             resize = transforms.Resize([313, 313], interpolation=transforms.InterpolationMode.BILINEAR)
#             db_spectrogram = resize(torchaudio.transforms.AmplitudeToDB()(mel_spectrogram).unsqueeze(0))
#             db_spectrogram = db_spectrogram.squeeze(0).squeeze(0)

#             mel_spec = db_spectrogram
#             spectrogram = torch.where(mel_spec == -np.inf, torch.zeros_like(mel_spec), mel_spec)
#             spectrogram = ((spectrogram - spectrogram.min()) /
#                            (spectrogram.max() - spectrogram.min()) * 255).numpy()

#             mean = np.mean((spectrogram))*1.5
#             print(np.mean(spectrogram))
#             # spectrogram[spectrogram < mean] = 0

#             cv2.imwrite(mel_path, spectrogram)


# def draw_mel(audio_path, index):
#     n_fft = 1024
#     hop_length = 512
#     signal, sr = torchaudio.load(audio_path)
#     mel_signal = librosa.feature.melspectrogram(y=signal.numpy(), sr=sr, hop_length=hop_length,
#                                                 n_fft=n_fft)
#     spectrogram = np.abs(mel_signal)
#     power_to_db = librosa.power_to_db(spectrogram, ref=np.max)

#     print(power_to_db.shape)

#     plt.figure(figsize=(8, 7))
#     librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma',
#                              hop_length=hop_length)
#     plt.colorbar(label='dB')
#     plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
#     plt.xlabel('Time', fontdict=dict(size=15))
#     plt.ylabel('Frequency', fontdict=dict(size=15))
#     plt.savefig(f"E:/code/SMIR/test/photo/{index}.png")


# paths = ['./datasets/val/wav_raw/val_raw_chunk1.wav',
#          ]
# draw_mel(paths[0], 1)
# draw_mel(paths[1], 2)

# wf, sr = torchaudio.load('./datasets/val/wav_raw/val_raw_chunk1.wav')

# # mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, win_length=1024,
# #                                                 hop_length=512, n_mels=128)(wf)

# print(wf.shape)

# print(type(wf))

# print(sr)


# print(mel_spec.shape)


# scale, sr = librosa.load('./datasets/val/wav_raw/val_raw_chunk1.wav')

# print(scale.shape)

# print(type(scale))

# print(sr)


# scale_file = 'E:/code/python/SMIR/datasets/val/wav_raw/val_raw_chunk100.wav'

# # load audio files with librosa
# scale, sr = librosa.load(scale_file)

# mel_spectrogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_fft=1024, hop_length=512, n_mels=128)

# mel_spectrogram.shape

# log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
# print(log_mel_spectrogram.shape)

# plt.figure(figsize=(20, 10))
# librosa.display.specshow(log_mel_spectrogram,
#                          x_axis="time",
#                          y_axis="mel",
#                          cmap='magma',
#                          sr=sr)
# plt.colorbar(format="%+2.f")
# plt.savefig("E:/code/python/SMIR/test/1.jpg")
# plt.close()

# scale_file = 'E:/code/python/SMIR/datasets/val/wav_record/val_record_chunk100.wav'

# # load audio files with librosa
# scale, sr = librosa.load(scale_file)

# mel_spectrogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_fft=1024, hop_length=512, n_mels=128)

# mel_spectrogram.shape

# log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
# print(log_mel_spectrogram.shape)

# plt.figure(figsize=(20, 10))
# librosa.display.specshow(log_mel_spectrogram,
#                          x_axis="time",
#                          y_axis="mel",
#                          cmap='magma',
#                          sr=sr)
# plt.colorbar(format="%+2.f")
# plt.savefig("E:/code/python/SMIR/test/2.jpg")
# plt.close()

# ir_paths = ["E:/code/python/SMIR/datasets/augmentation/ImpulseResponse/rir_1.wav",
#             "E:/code/python/SMIR/datasets/augmentation/ImpulseResponse/rir_2.wav",
#             "E:/code/python/SMIR/datasets/augmentation/ImpulseResponse/rir_3.wav",
#             "E:/code/python/SMIR/datasets/augmentation/ImpulseResponse/rir_4.wav"]

# bn_paths = ["E:/code/python/SMIR/datasets/augmentation/BackgroundNoise/bn-1.wav"]

# apply_augmentation = Compose(
#     transforms=[
#         # Gain(
#         #     min_gain_in_db=-25.0,
#         #     max_gain_in_db=-24.0,
#         #     p=1.0,
#         # ),
#         # AddColoredNoise(min_snr_in_db=2.8, max_snr_in_db=3.0, min_f_decay=0.9, max_f_decay=1.0, p=1.0)
#         # ApplyImpulseResponse(ir_paths, p=1.0),
#         # AddBackgroundNoise(bn_paths, p=1.0),
#         HighPassFilter(min_cutoff_freq=590,max_cutoff_freq=600,p=1.0),
#         LowPassFilter(min_cutoff_freq=4000,max_cutoff_freq=4001,p=1.0)
#         # HighPassFilter(p=1.0)
#     ]
# )

# waveform, sr = torchaudio.load('E:/code/python/SMIR/datasets/val/wav_record/val_record_chunk100.wav')
# waveform = apply_augmentation(waveform.unsqueeze(0), sample_rate=16000)
# torchaudio.save("E:/code/python/SMIR/test/aug.wav", waveform.squeeze(0), sr)
#
# scale_file = "E:/code/python/SMIR/test/aug.wav"
#
# # load audio files with librosa
# scale, sr = librosa.load(scale_file)
#
# mel_spectrogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
#
# mel_spectrogram.shape
#
# log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
# log_mel_spectrogram.shape
#
# plt.figure(figsize=(20, 10))
# librosa.display.specshow(log_mel_spectrogram,
#                          x_axis="time",
#                          y_axis="mel",
#                          cmap='magma',
#                          sr=sr)
# plt.colorbar(format="%+2.f")
# plt.savefig("E:/code/python/SMIR/test/3.jpg")
# # plt.show()
# plt.close()

# print("11111111111111111")
# waveform, sr = torchaudio.load("E:/code/python/SMIR/datasets/val/wav_raw/val_raw_chunk100.wav")

# waveform = apply_augmentation(waveform.unsqueeze(0), sample_rate=16000).squeeze(0)

# mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, win_length=1024,
#                                                        hop_length=512, n_mels=128)(waveform)

# resize = transforms.Resize([431, 431], interpolation=transforms.InterpolationMode.BILINEAR)
# db_spectrogram = resize(torchaudio.transforms.AmplitudeToDB()(mel_spectrogram).unsqueeze(0))
# # db_spectrogram = resize((mel_spectrogram).unsqueeze(0))
# db_spectrogram = db_spectrogram.squeeze(0).squeeze(0)
# print(db_spectrogram.shape)

# mel_spec = db_spectrogram
# # spectrogram = torch.where(mel_spec == -np.inf, torch.zeros_like(mel_spec), mel_spec)
# # print(spectrogram.shape)

# spectrogram = ((db_spectrogram - db_spectrogram.min()) /
#                (db_spectrogram.max() - db_spectrogram.min()) * 255).numpy().astype(np.uint8)

# # spec = spectrogram.transpose(1,2,0)
# print(spectrogram.shape)
# dst = cv2.applyColorMap(spectrogram, cv2.COLORMAP_MAGMA)
# print(dst.shape)
# print(type(dst))

# cv2.imshow('spec', dst)
# cv2.waitKey(0)


# rir_raw, sample_rate = torchaudio.load("E:/code/SMIR/datasets/augmentation/rir_raw_3.wav")
# rir = rir_raw[:, 0: 48000]
# rir = rir / torch.norm(rir, p=2)
# rir = torch.mean(rir, dim=0).unsqueeze(0)
# torchaudio.save("E:/code/SMIR/test/rir_3.wav",  rir, 16000)

# speech, _ = torchaudio.load("E:/code/SMIR/test/0001-01.wav")
# augmented = F.fftconvolve(speech, rir)

# torchaudio.save("E:/code/SMIR/test/0001-01-rir-3.wav", augmented, 16000)


# ir_paths = ["./datasets/augmentation/ImpulseResponse/rir_1.wav",
#             "./datasets/augmentation/ImpulseResponse/rir_2.wav"]

# bk_paths = ["./test/bk-2.wav"]

# with open("ir_inf.csv", 'w',newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow('old name','new name')
#     for name,id in zip(names_old,names_new):
#         writer.writerow([filename,name_new])


# import glob
# ir_paths = [path for path in glob.glob("./datasets/augmentation/ImpulseResponse/*")]
# bn_paths = [path for path in glob.glob("./datasets/augmentation/BackgroundNoise/*")]

# # ir_paths = ["E:/code/python/SMIR/datasets/augmentation/ImpulseResponse/rir_1.wav",
# #             "E:/code/python/SMIR/datasets/augmentation/ImpulseResponse/rir_2.wav",
# #             "E:/code/python/SMIR/datasets/augmentation/ImpulseResponse/rir_3.wav",
# #             "E:/code/python/SMIR/datasets/augmentation/ImpulseResponse/rir_4.wav"]

# # bn_paths = ["E:/code/python/SMIR/datasets/augmentation/BackgroundNoise/bn-1.wav"]

# apply_augmentation = Compose(
#     transforms=[
#         # Gain(
#         #     min_gain_in_db=-25.0,
#         #     max_gain_in_db=-24.0,
#         #     p=1.0,
#         # ),
#         # AddColoredNoise(min_snr_in_db=2.8, max_snr_in_db=3.0, min_f_decay=0.9, max_f_decay=1.0, p=1.0)
#         # ApplyImpulseResponse(ir_paths, p=1.0),
#         AddBackgroundNoise(bn_paths, p=1.0),
#         # HighPassFilter(min_cutoff_freq=590,max_cutoff_freq=600,p=1.0),
#         # LowPassFilter(min_cutoff_freq=4000,max_cutoff_freq=4001,p=1.0)
#         # HighPassFilter(p=1.0)
#     ]
# )

# waveform, sr = torchaudio.load('E:/code/python/SMIR/datasets/val/wav_raw/val_raw_chunk100.wav')

# for i in range(10):
#     waveform_ = apply_augmentation(waveform.unsqueeze(0), sample_rate=16000)
#     torchaudio.save(f"E:/code/python/SMIR/test/aug{i}.wav", waveform_.squeeze(0), sr)

# scale_file = "E:/code/python/SMIR/test/aug.wav"

# i=0
# for filename in os.listdir("./datasets/augmentation/ImpulseResponse"):
#     waveform,sr = torchaudio.load("./datasets/augmentation/ImpulseResponse/"+filename)
#     if sr == 16000:
#         i+=1

# print(i)


# class MusicData(Data.Dataset):
#     def init(self, paths, augment=False):
#         self.paths = paths  # a list of tuples containing the paths of anchor, positive and negative audio files
#         self.augment = augment  # a boolean flag indicating whether to apply data augmentation or not
#         self.mel_transform = torchaudio.transforms.MelSpectrogram(
#             sample_rate=16000, n_fft=1024, win_length=1024, hop_length=512, n_mels=128
#         )
#         self.image_transform = transforms.Compose(
#             [
#                 torchaudio.transforms.AmplitudeToDB(),
#                 transforms.Resize(
#                     [313, 313], interpolation=transforms.InterpolationMode.BILINEAR
#                 ),
#                 transforms.Lambda(lambda x: x[:, 50:288]),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5], [0.5]),
#                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#             ]
#         )


# def __getitem__(self, index):
#     a_path, p_path, n_path = self.paths[
#         index
#     ]  # get the paths of the audio files for the given index
#     a_waveform, _ = torchaudio.load_wav(
#         a_path
#     )  # load the anchor waveform as normalized tensor
#     p_waveform, _ = torchaudio.load_wav(
#         p_path
#     )  # load the positive waveform as normalized tensor
#     n_waveform, _ = torchaudio.load_wav(
#         n_path
#     )  # load the negative waveform as normalized tensor

#     if self.augment:
#         a_waveform = apply_augmentation(a_waveform.unsqueeze(0), sample_rate=16000).squeeze(0)  # apply some augmentation function to the anchor waveform if the flag is True

#     a_mel_spec = self.mel_transform(a_waveform)  # compute the mel spectrogram of the anchor waveform using torchaudio
#     p_mel_spec = self.mel_transform(p_waveform)  # compute the mel spectrogram of the positive waveform using torchaudio
#     n_mel_spec = self.mel_transform(n_waveform)  # compute the mel spectrogram of the negative waveform using torchaudio

#     a_img = self.image_transform(a_mel_spec)  # apply the image transform to the anchor spectrogram
#     p_img = self.image_transform(p_mel_spec)  # apply the image transform to the positive spectrogram
#     n_img = self.image_transform(n_mel_spec)  # apply the image transform to the negative spectrogram

#     return a_img, p_img, n_img  # return the images as tensors


# def __len__(self):
#     return len(self.paths)  # return the length of the dataset as the number of paths


# config = Config.from_json_file("config.json")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MusicNetModel(config.emd_size, config.class_nums).cuda()
# checkpoint = torch.load("./runs/train/exp/musicnet.pth")
# model.load_state_dict(checkpoint['model'],False)

# # 创建一个数据集和一个数据加载器
# val_file_path = "./datasets/val/wav"
# val_paths = generate_triplet(val_file_path)
# val_data = MusicData(val_paths, augment=False)
# val_data = Data.DataLoader(val_data, shuffle=False, batch_size=config.batch_size)
# val_nb = len(val_data)
# pbar_val = tqdm(val_data, total=val_nb)
# distances = []

# with torch.no_grad():
#     for step, (a_x, p_x, n_x) in enumerate(pbar_val):

#         a_x = a_x.cuda()
#         p_x = p_x.cuda()
#         n_x = n_x.cuda()

#         a_out, p_out, n_out = model(a_x), model(p_x), model(n_x)

#         s_d = F.pairwise_distance(a_out, p_out)
#         n_d = F.pairwise_distance(a_out, n_out)

#         dis = n_d - s_d
#         distances.extend(dis.tolist())

# array = np.array(distances)

# def classify(x):
#     if x < 0:
#         return "<0"
#     elif x >= 0 and x < 10:
#         return f"{int(x)}-{int(x)+1}"
#     else:
#         return ">10"

# categories = list(map(classify, array))
# unique_categories, counts = np.unique(categories, return_counts=True)

# def sort_key(x):
#     if x == "<0":
#         return -1
#     elif x == ">10":
#         return 11
#     else:
#         return int(x.split("-")[0])

# sorted_categories = sorted(unique_categories, key=sort_key)
# sorted_counts = [counts[unique_categories.tolist().index(c)] for c in sorted_categories]

# plt.bar(sorted_categories, sorted_counts)
# plt.xlabel("class")
# plt.ylabel("times")
# plt.title("Histogram of array element distribution")
# plt.savefig("./runs/train/exp/histogram.png")
# plt.show()


# index = []
# for i in range(len(distances)):
#     if distances[i] < 0 :
#         index.append(i)

# print(index)


# file = open("./distances.pkl", "wb")
# pickle.dump(list, file)
# file.close()

# file = open("./distances.pkl", "rb")
# list = pickle.load(file)
# file.close()


# def generate_random_number_positive(i):
#     random_number = i + random.randint(0, 4)
#     return random_number


# def generate_random_number_negative(i, num):
#     while True:
#         random_number = random.randint(1, num)
#         if random_number not in range(max(1, i - 41), i + 41):
#             return random_number

# train_file_path = "./datasets/val/wav_raw"
# train_paths = generate_triplet(train_file_path, "val")


# sound, sample_rate = torchaudio.load("G:/raw1.wav")
# sound_mono = torch.mean(sound, dim=0, keepdim=True)
# torchaudio.save("G:/raw.wav", sound_mono, sample_rate=sample_rate)


# for filename in os.listdir("G:/MyData/rirs_noises/rir_mono"):
#     file_path = "G:/MyData/rirs_noises/rir_mono/" + filename
#     waveform, sr = torchaudio.load(file_path)
#     resampler = torchaudio.transforms.Resample(sr, 16000)
#     waveform = resampler(waveform)
#     torchaudio.save(file_path, waveform, 16000)


# index = 162
# for filename in os.listdir("G:/MyData/rirs_noises/rir_mono"):
#     if filename[-4:] == ".wav":
#         file_path = "G:/MyData/rirs_noises/rir_mono/" + filename
#         output_path = f"./datasets/augmentation/ImpulseResponse/rir_{index}.wav"

#         rir_raw, sample_rate = torchaudio.load(file_path)
#         print(sample_rate)
#         rir = rir_raw[:, 0:48000]
#         rir = rir / torch.norm(rir, p=2)
#         print(rir.shape)
#         # rir = torch.mean(rir, dim=0).unsqueeze(0)
#         torchaudio.save(output_path, rir, 16000)
#         index += 1


# from pathlib import Path

# i = 0
# path_old = list()
# path_new = list()
# with open("G:/Data/fma_small/check.txt", "r") as f:
#     for line in f:
#         line = line.strip()
#         line = line.split("  ")
#         # print(line[1])
#         path_old.append(line[1])
#         path_o = "G:/Data/fma_small/" + line[1]
#         path_new.append(f"{i//1000+1:03d}/{i+1:04d}.mp3")
#         path_n = f"G:/Data/song_library_small/{i//1000+1:03d}/{i+1:04d}.mp3"
#         # print(path_o)
#         # print(path_n)

#         file = Path(path_o)
#         file.rename(path_n)
#         i += 1


# # print(path_new)
# # print(path_old)
# with open('G:/Data/song_library_small/song_inf.csv', 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['path_old', 'path_new'])
#         for name, id in zip(path_old, path_new):
#             writer.writerow([name, id])


# i = 0
# for i in range(10):
#     i += 1
#     index = f"{i:03d}"
#     print(index)
#     path = "G:/Data/song_library_small/" + index
#     os.mkdir(path)


# def random_list(x):
#     if x < 1000:
#         return None
#     else:
#         seq = range(1,x)
#         result = random.sample(seq, 1000)
#         return result

# import shutil
# # shutil.copyfile('file.py', 'file2.py')

# for i in random_list(106574):
#     print(i)
#     name = f"G:/Data/song_library/{i//1000+1:03d}/{i:06d}.mp3"

#     new= f"G:/Data/val_song/{i:06d}.mp3"
#     shutil.copyfile(name, new)


# def concatenate_audios(folder):
#     filenames = os.listdir(folder)

#     audio_files = [f for f in filenames if f.endswith(".mp3")]

#     result = AudioSegment.empty()

#     i = 0
#     for audio_file in audio_files:
#         if i >= 800:
#             audio = AudioSegment.from_mp3(folder + "/" + audio_file)

#             result += audio

#         i+=1
#         if i == 1000:
#             break

#     result.export("800-1000.wav", format="wav")

# concatenate_audios("G:/Data/song_library_small/001")


# file_path = "G:/Data/song_library_large"

# k = 0
# for i in range(1, 108):
#     file = file_path + f"/{i:03d}"
#     for filename in os.listdir(file):
#         if k > 107000:
#             file_name = file + "/" + filename
#             # print(file_name)
#             waveform, sr = torchaudio.load(file_name)
#             resampler = torchaudio.transforms.Resample(sr, 16000)
#             waveform = resampler(waveform)
#             waveform_mono = torch.mean(waveform, dim=0)
#             newfilename = "G:/code/SMIR/database/song" + "/" + filename[:-3] + "wav"
#             torchaudio.save(
#                 newfilename,
#                 waveform_mono.unsqueeze(0),
#                 sample_rate=16000,
#             )

#             if k % 200 == 0:
#                 print(file_name)

#         k += 1

#     print(i)


# sourcefile = AudioSegment.from_wav("C:/Users/syh/Downloads/Music/001154.wav")
# sourcefile.export("C:/Users/syh/Downloads/Music/001154.mp3", format="mp3")


# waveform, sr = torchaudio.load("C:/Users/syh/Downloads/Music/001682.wav")
# resampler = torchaudio.transforms.Resample(sr, 16000)
# waveform = resampler(waveform)
# torchaudio.save(
#                 "C:/Users/syh/Downloads/Music/001682-1.wav",
#                 waveform,
#                 sample_rate=16000,
#             )

# with open("G:/code/SMIR/database/duration.txt", "w") as f:
#     for filename in os.listdir("G:/code/SMIR/database/song"):
#         file_name = "G:/code/SMIR/database/song/" + filename
#         sound = AudioSegment.from_wav(file_name)
#         duration = sound.duration_seconds
#         if duration < 29:
#             print(file_name, duration)
#             f.write(f"{file_name} {duration}\n")
# print(duration)

# i+=1
# if i>10:
#     break


# with open("G:/code/SMIR/database/duration.txt", "r") as f:
#     for line in f:
#         line = f.readline()

#         line = line.strip()

#         line = line.split(" ")

#         if os.path.exists(line[0]):

#             os.remove(line[0])

#         print(line[0])






# step-1
# for filename in os.listdir("G:/MyData/processing/600-800"):
#     file_path = "G:/MyData/processing/600-800/" + filename
#     new_path = "G:/MyData/processing/600-800_wav/" + filename[:-3]+"wav"
#     sourcefile = AudioSegment.from_mp3(file_path)
#     sourcefile.export(new_path, format="wav")
#     waveform, sr = torchaudio.load(file_path)
#     resampler = torchaudio.transforms.Resample(sr, 16000)
#     waveform = resampler(waveform)
#     waveform_mono = torch.mean(waveform, dim=0)
#     torchaudio.save(new_path, waveform_mono.unsqueeze(0), sample_rate=16000)


# step-2
# waveform, sr = torchaudio.load("G:/600-800_record_1.wav")
# resampler = torchaudio.transforms.Resample(sr, 16000)
# waveform = resampler(waveform)
# torchaudio.save("G:/600-800_record_1.wav", waveform, 16000)


# step-3
# split = [0]
# sum = 0

# i = 0
# for filename in os.listdir("G:/MyData/processing/600-800_wav"):
#     filepath = "G:/MyData/processing/600-800_wav/" + filename

#     waveform, sr = torchaudio.load(filepath, normalize=True)
#     sum += len(waveform[0])
#     split.append(sum-1)

# print(split)


# waveform, sr = torchaudio.load("G:/600-800_record_1.wav")
# waveform = torch.mean(waveform, dim=0, keepdim=True)

# for i in range(len(split) - 1):
#     # 获取切割点的起始和结束位置
#     start = split[i]
#     end = split[i + 1]
#     # 按照第二个维度切割tensor，得到一个2*(end-start)的子tensor
#     slice = waveform[:, start:end]
#     filepath = f"G:/split-wav/song_{i+601}.wav"

#     torchaudio.save(filepath, slice, 16000)


# step-4
# df = pd.read_csv("G:/Data/song_library_small/song_inf.csv")

# i = 1
# for row in df.iterrows():
#     # num  = row[].lstrip("0")
#     if i >600:
#         num = row[1][1][4:-4].lstrip("0")
#         new_name = row[1][0][4:-4] + ".wav"


#         old_name = "G:/split-wav/song_" + num + ".wav"
#         new_name = "G:/split-wav/" + new_name
#         print(num)
#         print(row[1][0])
#         file = Path(old_name)
#         file.rename(new_name)

#     i += 1
#     if i > 800:
#         break

# name_map = dict()
# df = pd.read_csv("G:/Data/song_library_large/song_inf.csv")
# for row in df.iterrows():
#     # num  = row[].lstrip("0")
#     num = row[1][0][4:-4]
#     new_name = row[1][1][4:-4] + ".wav"
#     name_map[num] = new_name

# for filename in os.listdir("G:/split-wav"):
#     filenum  = filename[:-4]
#     newname = name_map[filenum]

#     old_name = "G:/split-wav/" + filename
#     new_name = "G:/split-wav/" + newname

#     file = Path(old_name)
#     file.rename(new_name)

# name_map = dict()
# df = pd.read_csv("./database/song_inf.csv")

# for row in df.iterrows():
#     # num  = row[].lstrip("0")
#     num = row[1][0][:-4]
#     new_name = row[1][1][:-4] + ".wav"
#     name_map[num] = new_name

# for filename in os.listdir("G:/split-wav"):
#     filenum  = filename[:-4]
#     if filenum in name_map:
#         newname = name_map[filenum]
#         old_name = "G:/split-wav/" + filename
#         new_name = "G:/split-wav/" + newname
#         file = Path(old_name)
#         file.rename(new_name)
    
#     else:
#         print(filename)


# step-5
# time = 20
# for filename in os.listdir("./runs/retrieval/test-30s"):

#     file_path= "./runs/retrieval/test-30s/" + filename
#     waveform, sample_rate = torchaudio.load (file_path)
#     length = time * sample_rate
#     waveform_trimmed = waveform [:, :length]
#     new_wav_file = "./runs/retrieval/test-20s/" + filename
#     torchaudio.save (new_wav_file, waveform_trimmed, sample_rate)


# arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.uint8)
# arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int32)
# a = transforms.ToTensor()(arr)




# def rename(file_path):
#     i = 1
#     for filename in os.listdir(file_path):
#         new_name = file_path+f"/{i:05d}.wav"
#         old_name = file_path + "/" + filename
#         file = Path(old_name)
#         file.rename(new_name)

#         i += 1

# rename("./datasets/mit/train/chunk-tensor-1-10000/data")



# config = Config.from_json_file("config.json")
# model = MusicNetModel(config.emd_size, config.class_nums).cuda()

# ir_paths = [path for path in glob.glob("./datasets/augmentation/ImpulseResponse/*")]
# bn_paths = [path for path in glob.glob("./datasets/augmentation/BackgroundNoise/*")]

# apply_augmentation = Compose(
#     transforms=[
#         Gain(
#             min_gain_in_db=-25.0,
#             max_gain_in_db=-20.0,
#             p=0.5,
#         ),
#         AddColoredNoise(min_snr_in_db=2.5, p=0.5),
#         ApplyImpulseResponse(ir_paths=ir_paths, p=0.5),
#         AddBackgroundNoise(background_paths=bn_paths, p=0.5),
#         HighPassFilter(p=0.5),
#         LowPassFilter(min_cutoff_freq=2400, p=0.5)
#     ]
# )

# def data_augmetation(input_file_path, output_file_path):
#     for filename in os.listdir(input_file_path):
#         file_path = input_file_path + "/" + filename
#         output_path = output_file_path + "/" + filename
#         waveform, sr = torchaudio.load(file_path, normalize=True)
#         waveform = waveform.unsqueeze(0)
#         waveform = apply_augmentation(waveform, sample_rate=sr)
#         waveform = waveform.squeeze(0)
#         torchaudio.save(output_path, waveform, sample_rate=sr)



# def get_tensor(input_file_path, output_file_path):
#     pbar = tqdm(os.listdir(input_file_path))
#     pbar.set_description('Processing:')
#     model.eval()

#     for filename in pbar:

#         wav_path = input_file_path + "/" + filename
#         waveform, sr = torchaudio.load(wav_path, normalize=True)
#         length = (len(waveform[0]) // sr) * sr
#         slices = range(0, length - 10 * sr + 1, sr)

#         for j, start in enumerate(slices):
#             end = slices[j] + 10 * sr
#             chunk = waveform[0][start:end].unsqueeze(0)

#             mel_spec = torchaudio.transforms.MelSpectrogram(
#                 sample_rate=sr, n_fft=1024, win_length=1024, hop_length=512, n_mels=128
#             )(chunk)

#             resize = transforms.Resize(
#                 [313, 313],
#                 interpolation=transforms.InterpolationMode.BILINEAR,
#                 antialias=True,
#             )
#             db_spec = resize(torchaudio.transforms.AmplitudeToDB()(mel_spec).unsqueeze(0))
#             db_spec = db_spec[:, :, 50:288, :].squeeze(0).squeeze(0)
#             mel_spec = (db_spec - db_spec.min()) / (db_spec.max() - db_spec.min()) * 255
#             mel_spec[mel_spec < 0.6 * mel_spec.mean()] = 0
#             mel_spec = mel_spec.numpy().astype(np.uint8)

#             img = cv2.applyColorMap(mel_spec, cv2.COLORMAP_MAGMA)
#             img = np.transpose(img, (2, 0, 1))
#             spec = torch.from_numpy(np.float32(img)).unsqueeze(0)

#             input_x = spec.to("cuda")
#             song_tag = filename[:-4]+f"-{j+1:02d}"
#             feature = model(input_x)
#             feature = feature.cpu().detach().numpy()

#             data_list = [song_tag]
#             for i in range(128):
#                 data_list.append(feature[0][i])
#             data = pd.DataFrame([data_list])
#             data.to_csv(output_file_path, mode="a", header=False, index=False)

#         pbar.update()




# checkpoint = torch.load("./runs/train/exp1-best/musicnet.pth")
# model.load_state_dict(checkpoint["model"], False)



# for i in range(5,10):

#     data_file = "./datasets/mit/train/chunk-tensor-1-10000/data"
#     csv_file = "./datasets/mit/train/chunk-tensor-1-10000/chunk-tensor.csv"

#     if i > 0:
#         data_file = f"./datasets/mit/train/chunk-tensor-1-10000/data_aug{i}"
#         csv_file = f"./datasets/mit/train/chunk-tensor-1-10000/chunk-tensor-aug{i}.csv"
#         os.mkdir(data_file)

#         data_augmetation("./datasets/mit/train/chunk-tensor-1-10000/data", data_file)

#     df_data = pd.DataFrame(columns=["song tag"] + [f"feature-{i}" for i in range(128)])
#     df_data.to_csv(csv_file, index=False)
#     get_tensor(data_file, csv_file)

#     if i > 0:
#         shutil.rmtree(data_file)





# df_data = pd.DataFrame(columns=["song tag"] + [f"feature-{i}" for i in range(128)])
# df_data.to_csv(f"./datasets/mit/test/chunk-tensor-raw.csv", index=False)
# get_tensor(f"./datasets/mit/test/test-raw", f"./datasets/mit/test/chunk-tensor-raw.csv")






# data_raw = load_data("./datasets/mit/test/chunk-tensor-raw.csv")
# with shelve.open("./database/chunk-tensor-db/chunk-tensor", "c") as db:
#     for key, value in data_raw.items():
#         db[key] = value
# i=0
# with shelve.open("./database/chunk-tensor-db/chunk-tensor", "r") as db:
#     for key, value in db.items():
#         print(key, value)
#         i+=1
#         if i > 10:
#             break
# with open("./database/chunk-tensor.pkl", "wb") as f:
#     pickle.dump(data_raw, f)

# with open("./database/chunk-tensor.pkl", "rb") as f:
#     data = pickle.load(f)



# a_list = [[1,2,3],[3,5,6],[11,2,2.5]]
# torch.save(a_list, './test/test.pt')

# b_list = torch.load('./test/test.pt')
# print(b_list)


# with shelve.open("./database/chunk-tensor-db/chunk-tensor", "r") as db:
#         for key, value in tqdm.tqdm(db.items()):
#             file_path = f"./test/chunk-tensor-all/{key}.pt"
#             torch.save(value, file_path)




# file = "./datasets/mit/train/chunk-tensor-len.pt"
# a_list = []
# for i in range(1, 105373):
#     if os.path.exists(f"./datasets/mit/train/chunk-tensor-all/{i:06d}.pt") is False:
#         a_list.append(0)
#     else:
#         k = torch.load(f"./datasets/mit/train/chunk-tensor-all/{i:06d}.pt")
#         a_list.append(len(k))

#     if i< 10:
#         print(a_list)

# torch.save(a_list, file)



# class SpectrogramFingerprintData(Data.Dataset):
#     def __init__(self, triplets, mode):
#         self.triplets = triplets
#         self.mode = mode

#     def __getitem__(self, index):       
#         a_part, p_part, n_part, id = self.triplets[index]       

#         a_seq = np.zeros((50, 128))
#         p_seq = np.zeros((50, 128))
#         n_seq = np.zeros((50, 128))
 
#         if self.mode == "train":
#             data = train_data_raw
#             n_part[0] = generate_random_number_negative(a_part[0], 105372)
            
#             while data_len[n_part[0] - 1] < 20:
#                 n_part[0] = generate_random_number_negative(a_part[0], 105372)
            
#             n_sample = torch.load(f"./datasets/mit/train/chunk-tensor-all/{n_part[0]:06d}.pt")
#             n_part[2] = len(n_sample)

#         elif self.mode == "val":
#             data = val_data_raw
#             id = 1
#             a_part[1] = 0
#             a_part[2] = len(data[0][a_part[0]])

#         for i in range(a_part[1], min(a_part[2] + 1, 49)):
#             a_seq[i - a_part[1]] = data[id][a_part[0]][i - 1]
#         for i in range(p_part[1], min(p_part[2] + 1, 49)):
#             p_seq[i - p_part[1]] = data[0][p_part[0]][i - 1]
#         for i in range(n_part[1], min(n_part[2] + 1, 49)):
#             if self.mode == "train":
#                 n_seq[i - n_part[1]] = n_sample[i - 1]
#             elif self.mode == "val":
#                 n_seq[i - n_part[1]] = data[0][n_part[0]][i - 1]

#         return np.float32(a_seq), np.float32(p_seq), np.float32(n_seq)

#     def __len__(self):
#         return len(self.triplets)


# import shutil
# file = "F:/gtzan"
# k = 1

# for filename in os.listdir(file):
#     new_file = file +"/" + filename
#     for filename2 in os.listdir(new_file):
#         new_name = f"F:/gtzan_/{k:04d}.wav"
#         old_name = new_file + "/" + filename2
#         shutil.copy(old_name , new_name)
#         k+=1


# waveform, sample_rate = torchaudio.load("./runs/retrieval/test_1_volume=75.mp3", normalize=True)
# resampler = torchaudio.transforms.Resample(sample_rate, 16000)
# waveform = resampler(waveform)
#
# torchaudio.save("./runs/retrieval/test_1_volume=75.wav", waveform, 16000)


# file = "F:/gtzan_"

# for filename in os.listdir(file):
#     new_file = file +"/" + filename
#     waveform, sample_rate = torchaudio.load(new_file, normalize=True)
#     resampler = torchaudio.transforms.Resample(sample_rate, 16000)
#     waveform = resampler(waveform)
#     torchaudio.save(new_file, waveform, 16000)


# for i in range(10, 13):

#     data_file = "./datasets/mit/train/chunk-tensor-1-10000/data"
#     csv_file = "./datasets/mit/train/chunk-tensor-1-10000/chunk-tensor.csv"

#     if i > 0:
#         data_file = f"./datasets/mit/train/chunk-tensor-1-10000/data_aug{i}"
#         csv_file = f"./datasets/mit/train/chunk-tensor-1-10000/chunk-tensor-aug{i}.csv"
#         os.mkdir(data_file)

#         data_augmetation("./datasets/mit/train/chunk-tensor-1-10000/data", data_file)

#     df_data = pd.DataFrame(columns=["song tag"] + [f"feature-{i}" for i in range(128)])
#     df_data.to_csv(csv_file, index=False)
#     get_tensor(data_file, csv_file)

#     if i > 0:
#         shutil.rmtree(data_file)



# data_map = load_data("F:/gtzan_aug/chunk-tensor.csv")

# for key,value in tqdm(data_map.items()):
#     name = f"F:/gtzan_aug/data1/{key}-00.pt"
#     torch.save(value, name)

# for i in range(22,23):
#     data_map_aug = load_data(f"F:/gtzan_aug/chunk-tensor-aug{i}.csv")
#     for key,value in tqdm(data_map_aug.items()):
#         name = f"F:/gtzan_aug/data1/{key}-{i-3:02d}.pt"
#         torch.save(value, name)



# for i in range(13, 23):

#     data_file = "F:/gtzan_aug/data"
#     csv_file = "F:/gtzan_aug/chunk-tensor.csv"

#     if i > 0:
#         data_file = f"F:/gtzan_aug/data_aug{i}"
#         csv_file = f"F:/gtzan_aug/chunk-tensor-aug{i}.csv"
#         os.mkdir(data_file)

#         data_augmetation("F:/gtzan_aug/data", data_file)

#     df_data = pd.DataFrame(columns=["song tag"] + [f"feature-{i}" for i in range(128)])
#     df_data.to_csv(csv_file, index=False)
#     get_tensor(data_file, csv_file)

#     if i > 0:
#         shutil.rmtree(data_file)

# data_augmetation("./test/wav","./test/wav_aug")

# data_augmetation("./datasets/mit/train/chunk-tensor-1-10000/data1","./datasets/mit/train/chunk-tensor-1-10000/data1-aug")




# li = [51.0827,47.3400,50.4018,48.3613,46.3207,52.1834,47.8595,48.0142]

# sum = 0
# for i in range(len(li)):
#     if i == 0:
#         sum += li[i]*10
#     else:
#         sum += li[i]

# print(sum/8)

# tag = np.zeros(10001)
# # print(tag)

# for filename in os.listdir("./datasets/mit/train/data"):
#     num = int(filename[:-6])
#     tag[num] = 1


# nums = []
# for i in range(1,10001):
#     if tag[i] == 0:
#         nums.append(i)

# print(nums)

# [544, 548, 3488, 4114, 4138, 4457, 4458, 4459, 4460, 4461, 4462, 4463, 4464, 4465, 4466, 4467, 4468, 4469, 4470, 4471, 4472, 4473, 4474, 4475, 4476, 4477, 4478, 4479, 4480, 4481, 4482, 4483, 4484, 4485, 4486, 4487, 4488, 4489, 4490, 4491, 4492, 4493, 4494, 4495, 4496, 4497, 4498, 4499, 4500, 4501, 4502, 4503, 4504, 4505, 4506, 4507, 4508, 4509, 4510, 4511, 4514, 4515, 4516, 4517, 4518, 4519, 4520, 4521, 4522, 4523, 4524, 4525, 4526, 4527, 4528, 4529, 4530, 4531, 4532, 4647, 4660, 4669, 5292, 5368, 5392, 5476, 5479, 5482, 5484, 5486, 5489, 5492, 5495, 5499, 5502, 5735, 6333, 6588, 6593, 6754, 7941, 9177, 9824]

# import torch

# a = torch.tensor([1,2,8,4,5,6,7,3])
# b = torch.rand((8, 3))
# print(b)
# print(b[a-1])






# import numpy as np
# import os

# a = 0
# b = [0]
# i = 0
# k = np.zeros(500,dtype=int)
# for filename in os.listdir("./runs/retrieval/test/db-ts-aug"):
#     file_path = "./runs/retrieval/test/db-ts-aug/" + filename
#     m = np.load(file_path)
#     a = a + m.shape[0]
#     b.append(a)
#     k[i+1] = a + 1
#     i+=1
#     if i == 499:
#         break

# print(k)
# np.save("./test_ids_icassp2021.npy", k)


# import numpy as np
# import os
# from tqdm import tqdm

# a = np.load("./database/db_inf.npy")
# # # print(a[:-10])

# # # print(a.shape)



# sum = 0
# for i in range(a.shape[0]):
#     sum+=a[i][1]

# print(sum)

# arr_shape = (6131510, 128)
# arr = np.memmap("./database/raw_db.mm",
#                 dtype='float32',
#                 mode='w+',
#                 shape=arr_shape)
# np.save("./database/raw_db_shape.npy", arr_shape)

# k = 0

# for i, filename in enumerate(tqdm(os.listdir("./database/db"))):
#     vector = np.load("./database/db/"+ filename)
#     arr[k:k+a[i][1], :] = vector
#     k = k + a[i][1]

# # print(f"=== Succesfully stored {arr_shape[0]} fingerprint to {os.path.dirname(output_root_dir)} ===")
# arr.flush(); 

# a_shape = np.load("./runs/retrieval/test/300_shape.npy")
# a = np.memmap("./runs/retrieval/test/300.mm", dtype='float32', mode='r+', shape=(a_shape[0], a_shape[1]))


# print(type(a[0][0]))
# print(a[0][0])



# # b = np.load("./runs/retrieval/test/db/000134.npy")
# b_shape = np.load("./runs/retrieval/test/500_shape.npy")
# b = np.memmap("./runs/retrieval/test/500.mm", dtype='float32', mode='r+', shape=(a_shape[0], a_shape[1]))
# print(type(b[0][0]))
# print(b[0][0])

# # print(arr[0:100, : ])
# print(arr[0:1, : ])
# print(arr[:-2, : ])


# del(arr)
# print(k)



# print(6131510-29500)
# sum = 0
# for filename in os.listdir("./database/db"):
#     file = "./database/db/" + filename
#     k = np.load(file)
#     sum+= len(k)

# print(sum)
# a = np.zeros(500).astype(int)
# sum1 = 0
# for ti, filename in enumerate(os.listdir("./runs/retrieval/test/query")):
#     file = "./runs/retrieval/test/query/" + filename
#     k = np.load(file)
#     sum1+= len(k)
#     if ti < 499:
#         a[ti+1] = sum1
    
# np.save("./test_ids_icassp2021.npy", a)


# print(sum1)
# print(sum-sum1)


# import random

# k = 0
# for i in range(10000):
#     random_float = random.random()**6
#     if random_float > 0.2:
#         k+=1

# print(k/10000)

# m = np.load("D:/neural-audio-fp-dataset/music/test-query-db-500-30s/seg_list.npy")
# print(m.shape)

# print(6102010+29500)


# a = np.zeros((105363,2)).astype(int)
# for ti, filename in enumerate(tqdm(os.listdir("./database/db"))):
#     num = int(filename[:-4])
#     a[ti][0] = num
#     a[ti][1] = len(np.load("./database/db/"+filename))

# np.save("./database/db_inf.npy", a)

# a = np.load("./runs/retrieval/test/db/000134.npy")

# norm= np.linalg.norm(a[5])

# print(norm)



# data_shape = np.load("./database/raw_db_shape.npy")
# data1 = np.memmap("./database/raw_db.mm", dtype='float32', mode='r+',
#                         shape=(data_shape[0], data_shape[1]))

# print(data1[0][0])



# a = np.load("./aaa.npy")
# b = np.load("./bbb.npy")

# print(a.shape)

# for i in range(500):
#     for j in range(6):
#         if a[i][j] != b[i][j]:
#             print(i,j)

# a = np.random.random((2, 3))
# a_mean = np.mean(a, axis = 0)
# b = np.zeros((2,3))
# b[0] = a_mean
# print(b)
# print(a)
# print(a_mean)


# import subprocess

# command2 = "C:/Users/syh/.conda/envs/SMIR/python.exe g:/code/SMIR/train_arcface.py --model mt --lr 0.001 --epoch 30"
# subprocess.call(command2, shell=False)


# command1 = "C:/Users/syh/.conda/envs/SMIR/python.exe g:/code/SMIR/train_arcface.py --model mt --lr 0.0001 --epoch 60"
# subprocess.call(command1, shell=False)

# import random

# t = np.zeros(100)
# for i in range(1000):
#     x = random.random()
#     if x <= 0.9:
#         y = x / 9
#     else:
#         y = 9 * x - 8
#     chunk_len = min(max(int(y * 59), 1), 59)
#     t[chunk_len - 1]+=1

# print(t[:10])

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
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm
import logging
from losses.ArcFace import ArcFaceLoss, ArcFaceLoss_new
import argparse
import faiss
from evaluate import RetrievalData
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses.Tam import TamLoss

# model = MT(
#             input_dim=128,
#             output_dim=128,
#             dim=256,
#             depth=8,
#             heads=32,
#             dim_head=16,
#             mlp_dim=256,
#             dropout=0.1,
#         ).cuda()

# arcface = TamLoss(in_features=128, out_features=20000, s=64.0).cuda()


# checkpoint = torch.load("./runs/checkpoint/mt_arcface/exp-dl-5-/mt_17.pth")
# model.load_state_dict(checkpoint["model"])
# arcface.load_state_dict(checkpoint["arcface"])
# start_epoch = checkpoint["epoch"]

# model.eval()
# for i in range(2, 3):
#     # filename = f"./datasets/mt/val/data/{i}-00.npy"
#     filename = f"./datasets/mt/train/data/{i}-03.npy"
#     filename1 = f"./datasets/mt/train/data/{i}-00.npy"
#     b = np.load(filename)
#     c = np.load(filename1)
#     print(len(b))
#     print(len(c))

#     t = np.zeros((64, 128)).astype(np.float32)
#     for j in range(1):
#         t[j] = b[j]

#     tensor = torch.from_numpy(t).cuda()
#     tensor = tensor.unsqueeze(0)
#     feature = model(tensor)
#     a = arcface.weight[i + 10000]
#     a = a.unsqueeze(0)
#     cos = F.cosine_similarity(feature, a)
#     print(cos)

#     t = np.zeros((64, 128)).astype(np.float32)
#     for j in range(5):
#         t[j] = b[j]

#     tensor = torch.from_numpy(t).cuda()
#     tensor = tensor.unsqueeze(0)
#     feature = model(tensor)
#     a = arcface.weight[i + 10000]
#     a = a.unsqueeze(0)
#     cos = F.cosine_similarity(feature, a)
#     print(cos)


#     t = np.zeros((64, 128)).astype(np.float32)
#     for j in range(len(b)):
#         t[j] = b[j]

#     tensor = torch.from_numpy(t).cuda()
#     tensor = tensor.unsqueeze(0)
#     feature = model(tensor)
#     a = arcface.weight[i + 10000]
#     a = a.unsqueeze(0)
#     cos = F.cosine_similarity(feature, a)
#     print(cos)


#     t = np.zeros((64, 128)).astype(np.float32)
#     for j in range(len(b)):
#         t[j] = c[j]

#     tensor = torch.from_numpy(t).cuda()
#     tensor = tensor.unsqueeze(0)
#     feature = model(tensor)
#     a = arcface.weight[i + 10000]
#     a = a.unsqueeze(0)
#     cos = F.cosine_similarity(feature, a)
#     print(cos)

# a = np.load("./center.npy")
# print(a[0])


# chunk_len = 7
# start = 2
# # a = np.random.randn(30)
# # b = np.zeros(15)
# # for i in range(15):
# #     b[i] = a[start + (i%chunk_len)]

# # print(a)
# # print(b)
# print(chunk_len//start)

# length_all = dict()
# maxlen = 0
# na = ""
# for filename in tqdm(os.listdir("./datasets/mt/train/data")):
#     num = int(filename.split("-")[1][:-4])
#     # if num >= 10:
#     #     os.remove("./datasets/mt/train/data/" + filename)

#     if num == 0:
#         a = np.load("./datasets/mt/train/data/" + filename)
#         length = len(a)

#         # if length == 6:
#         #     na = filename          
#         #     print(na)
#         if length not in length_all.keys():
#             length_all[length] = 1
#         else:
#             length_all[length] += 1
   

# print(length_all) 







# for filename in os.listdir("./datasets/mt/train/111"):
#     k = np.load("./datasets/mt/train/111/" + filename)   
#     for t in range(327//59):
#         a = np.zeros((59, 128), dtype=np.float32)
#         n = 59*t
#         for i in range(59):
#             a[i] = k[i + n]
#         np.save(f"./datasets/mt/train/new/{t:02d}-" + filename.split('-')[1][:-4] + ".npy", a)



# n = np.load("./datasets/mt/train/new/01-00.npy")

# print(type(n[0][0]))



# def renameFile():
#     filenum = 4
#     new_name = 2676


#     for filename in os.listdir("./datasets/mt/train/new"):
#         if int(filename.split('-')[0]) == filenum:
#             old = "./datasets/mt/train/new/" + filename
#             new = "./datasets/mt/train/new/" + str(new_name) + f"-{filename.split('-')[1]}"
#             os.rename(old, new)

# renameFile()

import shutil

# for filename in tqdm(os.listdir("./datasets/mt/train/data")):
#     num = int(filename.split('-')[1][:-4])
#     if num <= 4 or num >= 15:
#         new_path = "./datasets/mt/train/data-new/" + filename
#         old_path = "./datasets/mt/train/data/" + filename
#         shutil.copy(old_path, new_path)

# for filename in tqdm(os.listdir("./datasets/mt/train/data-new")):
#     num = int(filename.split('-')[1][:-4])
#     if num >= 10:
#         old_path = "./datasets/mt/train/data-new/" + filename
#         new_path = "./datasets/mt/train/data-new/" + filename.split('-')[0] + f"-{num-10:02d}.npy"
#         os.rename(old_path, new_path)



# a = np.load("./datasets/mt/train/data-1k/1-00.npy")
# print(a[0])


# import torch
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# data = np.load(f"./datasets/mt/train/data-new/1-00.npy")
# # print(data.shape)
# # t_c = torch.from_numpy(data).float()
# # print(t_c.shape)
# # print(t_c.unsqueeze(0).shape)

# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data)

# pca = PCA(n_components=2)
# reduce_data = pca.fit_transform(scaled_data).reshape(-1)

# # print(reduce_data.reshape(-1))

# a = np.zeros((2, 128))

# a[0][:len(reduce_data)] = reduce_data[:]
# print(a)

# k = 0
# for filename in tqdm(os.listdir("./datasets/mt/train/data-15s")):
#     # num = int(filename.split('-')[0])
#     # other_part = filename.split('-')[1]
#     # a = np.load("./datasets/mt/train/data/" + filename)
#     # mid = len(a)//2

#     # front_half = a[:mid]
#     # back_half = a[mid:]

#     # np.save(f"./datasets/mt/train/data-15s/{num}" + other_part, front_half)
#     # np.save(f"./datasets/mt/train/data-15s/{num+20000}" + other_part, back_half)
#     filename_1 = filename[:-6]
#     filename_2 = filename[-6:]
#     new_name = filename_1 + '-' + filename_2
#     old_path = "./datasets/mt/train/data-15s/" + filename
#     new_path = "./datasets/mt/train/data-15s/" + new_name
#     os.rename(old_path, new_path)

# data = np.memmap("./runs/retrieval/test/query.mm", dtype='float32', mode='r+')

# print(data[:128])




import csv
import random
import shutil
import pandas as pd
import torchaudio
import os
import ast
from tqdm import tqdm
# from losses import TripletLoss, ArcFaceLoss_center
import torch
import torch.nn.functional as F
import numpy as np
import faiss
import matplotlib.pyplot as plt
import pickle


# def wav2mono_8000(input_path, output_path):
#     waveform, sample_rate = torchaudio.load(input_path, normalize=True)
#     resampler = torchaudio.transforms.Resample(sample_rate, 8000)
#     waveform = resampler(waveform)
#     torchaudio.save(output_path, waveform, 8000)

# for filename in tqdm.tqdm(os.listdir("./database/song")):
#     wav2mono_8000("./database/song/" + filename, "E:/neural-audio-fp-main/neural-audio-fp-dataset/music/test-dummy-db-100k-full/from_fma_large100k-30s/" + filename)

# for filename1 in os.listdir("F:\Data\song_library_small"):
#     for filename2 in os.listdir("F:\Data\song_library_small\\" + filename1):
#         wav2mono_8000("F:\Data\song_library_small\\" + filename1 + "\\" + filename2, "F:\Data\song\\" + filename2[:-4] + ".wav")


# k = 0
# for filename in os.listdir("F:/MyData/new-dataset/song-16k"):
#     input_path = "F:/MyData/new-dataset/song-16k/" + filename
#     waveform, sr = torchaudio.load(input_path, normalize=True)
#     length = (len(waveform[0]) // sr) * sr
#     slices = range(0, length - 40000 + 1, 40000)
#     for j, start in enumerate(slices):
#         end = slices[j] + 40000
#         chunk = waveform[0][start:end].unsqueeze(0)
#         torchaudio.save(f"./datasets/val_v2/wav_raw/val_chunk_{k}.wav", chunk, sample_rate=16000)
#         k += 1

# for filename in os.listdir("F:/111"):
#     wav2mono_16000("F:/111/" + filename, "F:/222/" + filename[:-4] + ".wav")

# split = [0]
# sum = 0

# i = 0
# for filename in os.listdir("F:/MyData/new-dataset/song-16k"):
#     filepath = "F:/MyData/new-dataset/song-16k/" + filename

#     waveform, sr = torchaudio.load(filepath, normalize=True)
#     # print(len(waveform[0]))
#     # print(sr)
#     sum += len(waveform[0])
#     split.append(sum-1)

# print(split)


# waveform, sr = torchaudio.load("F:/new_out.wav")

# for i in range(len(split) - 1):
#     # 获取切割点的起始和结束位置
#     start = split[i]
#     end = split[i + 1]
#     # 按照第二个维度切割tensor，得到一个2*(end-start)的子tensor
#     slice = waveform[:, start:end]
#     filepath = f"F:/split-wav-out/song_{i+1}.wav"

#     torchaudio.save(filepath, slice, 16000)

# import glob
# from torch_audiomentations import (
#     Compose,
#     Gain,
#     AddColoredNoise,
#     ApplyImpulseResponse,
#     AddBackgroundNoise,
#     LowPassFilter,
#     HighPassFilter,
# )
# import torchaudio
# import cv2
# import ast
# from losses import TripletLoss


# ir_paths = [path for path in glob.glob("./datasets/augmentation/ImpulseResponse/*")]
# bn_paths = [path for path in glob.glob("./datasets/augmentation/BackgroundNoise/*")]

# apply_augmentation = Compose(
#     transforms=[
#         Gain(
#             min_gain_in_db=-25.0,
#             max_gain_in_db=-20.0,
#             p=0.5,
#         ),
#         AddColoredNoise(min_snr_in_db=10.0, p=0.5),
#         ApplyImpulseResponse(ir_paths=ir_paths, p=0.5),
#         AddBackgroundNoise(min_snr_in_db=5.0, background_paths=bn_paths, p=0.5),
#         # HighPassFilter(p=0.5),
#         # LowPassFilter(min_cutoff_freq=2400, p=0.5),
#         # PitchShift(
#         #     min_transpose_semitones=-1.0,
#         #     max_transpose_semitones=1.0,
#         #     p=0.2,
#         #     sample_rate=16000,
#         # ),
#     ]
# )
    
# for filename in os.listdir("./runs/retrieval/wav_aug"):
#     file_path = "./runs/retrieval/wav_aug/" + filename
#     waveform, sr = torchaudio.load(file_path)
#     waveform = apply_augmentation(waveform.unsqueeze(0), sample_rate=16000).squeeze(0)
#     torchaudio.save(file_path, waveform, 16000)


# import os
# import librosa
# import soundfile as sf
 
 
# def resample4wavs(frompath, topath, resamplerate):
#     '''
#     :param frompath: 源文件所在目录
#     :param topath: 重置采样率文件存放目录
#     :param resamplerate: 重置采样率
#     :return:
#     '''
#     fs = os.listdir(frompath)
#     for f in tqdm.tqdm(fs):
#         try:
#             fromfile = frompath + f
#             # print(fromfile)
#             tofile = topath + f
#             y, sr = librosa.load(fromfile)
#             to_y = librosa.resample(y, orig_sr=sr, target_sr=resamplerate)
#             # librosa.output.write_wav(tofile, to_y, resamplerate)过时代码, 需要换成下面的代码
#             sf.write(tofile, to_y, resamplerate)
#         except Exception as e:
#             print('Error:', e)




# import torchaudio
# import torch

# for i in tqdm.tqdm(range(1, 10001)):
#     random_numbers = random.sample(range(1, 10001), 10)
#     waveforms = []
#     lens = []
#     for j in range(10):
#         fromfile = f"E:\\neural-audio-fp-main\\neural-audio-fp-dataset\\music\\train\\{random_numbers[j]:05d}.wav"
#         y, sr = torchaudio.load(fromfile)
#         waveforms.append(y)
#         lens.append(y.shape[1])

#     len = min(lens)
#     waveform = torch.zeros(1, len)
#     index = 0
#     while index < len:
#         lc = random.randint(2000,4000)
#         k = random.randint(0,9)
#         if index + lc >= len:
#             waveform[:,index:len] = waveforms[k][:,index:len]
#         else:
#             waveform[:,index:index+lc] = waveforms[k][:,index:index+lc]
#         index += lc

#     outfile = f"E:\\neural-audio-fp-main\\neural-audio-fp-dataset\\music\\train_generate\\{i+10000:05d}.wav"
#     torchaudio.save(outfile,waveform, 8000)

# import torch
# a = torch.ones(5)

# print(a[1:1])





        # sf.write(tofile, to_y, 8000)


# resample4wavs("E:/neural-audio-fp-main/neural-audio-fp-dataset/music/train_generate/", "E:/neural-audio-fp-main/neural-audio-fp-dataset/music/train_new/", 8000)



# import shutil
# for filename in os.listdir("./datasets/mit/train/data_20"):
#     if int(filename.split('-')[1][:-4]) >= 10:
#         os.remove("./datasets/mit/train/data_20/"+filename)


# for filename in os.listdir("./datasets/mit/val/data"):
#     shutil.copy("./datasets/mit/val/data/"+filename, "./datasets/mit/val/data_20/"+filename)

# for filename in os.listdir("./datasets/mit/val/data"):
#     if int(filename.split('-')[1][:-4]) == 0:
#         new_name = f"{int(filename.split('-')[0]):05d}.npy"
#         shutil.copy("./datasets/mit/val/data/"+filename, "./datasets/mit/val/val_db/"+new_name)

# import numpy as np
# data = np.load("./datasets/mit/train/data/1-00.npy")
# sequence = np.zeros((3, 64, 128))
# for i in range(0, min(len(data), 64)):
#     sequence[0][i] = data[i]
#     sequence[1][i] = data[i]
#     sequence[2][i] = data[i]

# print(sequence)

# import numpy as np
# db_index = np.load("./database/db_inf.npy")

# print(db_index[:10])




# def retrieval_new():
#     T1 = time.time()
#     model_mit.eval()
#     db_shape = np.load("./database/db_shape.npy")
#     db = np.memmap("./database/db.mm", dtype='float32', mode='r+', shape=(db_shape[0], db_shape[1]))
#     query_shape = np.load("./datasets/mit/val/val_db_shape.npy")
#     query = np.memmap("./datasets/mit/val/val_db.mm", dtype='float32', mode='r+', shape=(query_shape[0], query_shape[1]))

#     db_index = np.load("./database/db_inf.npy")
#     db_index_map = dict()
#     for i in range(len(db_index)):
#         db_index_map[db_index[i][0]] = i


#     with open("./datasets/mit/val/val_song_inf.csv", "r") as val_inf_file:
#         val_inf = list(csv.reader(val_inf_file))
#         val_inf = np.array(val_inf)

#     # print(val_inf[1])

#     val_dict = dict()
#     for i in range(1,len(val_inf)):
#         val_dict[val_inf[i][1]] = val_inf[i][0]

#     xb_len = db_shape[0]
#     xb = np.zeros((xb_len, 256))
#     for i in range(xb_len):
#             xb[i] = db[i]

#     index = faiss.IndexFlatIP(256)
#     index.add(xb)

#     correct_num_1 = 0
#     correct_num_10= 0
#     filenames = os.listdir("./datasets/mit/val/val_db")
#     for ti in range(query_shape[0]):
#         # qi = int(filenames[ti][:-4])
#         qi = int(val_dict[filenames[ti][:-4]])
#         # print(qi)
#         if qi == 10441 or qi == 39377 or qi == 98566:
#             continue
#         q = np.zeros((1, 256))
#         q[0] = query[ti]
#         D, I = index.search(q, 10)

#         if db_index_map[qi] == I[0][0]:
#             correct_num_1 += 1
#             correct_num_10 += 1
#         elif db_index_map[qi] in I[0]:
#             correct_num_10 += 1
#         # else:
#         #     print(qi, db_index_map[qi], I)

#     T2 = time.time()
#     print("程序运行时间:%s毫秒" % ((T2 - T1) * 1000))
#     print("top1 accuracy:", correct_num_1 / query_shape[0])
#     print("top10 accuracy:", correct_num_10 / query_shape[0])
#     print(query_shape[0], correct_num_1)




# import shutil
# for filename in os.listdir("./datasets/mit/train/data-old"):
#     if int(filename.split('-')[0]) < 10000:
#         shutil.copy("./datasets/mit/train/data-old/"+filename, "./datasets/mit/train/data-0-1w/"+filename)

# import numpy as np
# data = np.random.rand(100, 3)
# sequence = np.zeros((100, 3))
# sequence[:10] = data[:10]

# print(data)
# print(sequence)
# print(sequence.shape)




# k = np.random.rand(6, 3)
# k_1 = np.random.rand(5, 6)

# # np.concatenate((avg, avg), axis=1)
# for i in range(len(k)-1):
#     k_1[i] = np.concatenate((k[i], k[i+1]), axis=0)

# mean = np.mean(k_1, axis=0)

# print(k)
# print(k_1)
# print(mean)


# max = 0
# min = 100
# min_index = 0
# length = []
# for i in range(1, 20001):
#     data = np.load(f"./datasets/mit/train/data-new/{i}-00.npy")
#     if data.shape[0] > max:
#         max = data.shape[0]
#     if data.shape[0] < min:
#         min = data.shape[0]
#         min_index = i
#     length.append(data.shape[0])


# print(max, min_index, min)

# plt.hist(length, bins=10, edgecolor='black')

# # 添加标题和轴标签
# plt.title('Data Distribution')
# plt.xlabel('Value')
# plt.ylabel('Frequency')

# # 显示图形
# plt.show()


# np.concatenate((avg, avg), axis=1)







# arcface = ArcFaceLoss_center(in_features=256, out_features=20000, s=64.0, m=0.25)
# print(arcface.weight[0][0].dtype)
# print(arcface.weight.shape)

# d = np.load("./datasets/mit/train/data-new/1-00.npy")
# # m = np.zeros((len(d)-1, 256))
# # for j in range(len(data)-1):
# #         m[j] = np.concatenate((d[j], d[j+1]), axis=0)

# # d_m = np.mean(m, axis=0)
# # print(d_m.shape)
# # print(d_m)
# print(d[0][0].dtype)


# for filename in os.listdir("./database/db"):
#     filepath = "./database/db/" + filename
#     data = np.load(filepath)
#     if len(data) <= 1:
#         print(filename)

# a = np.load("./database/db_inf.npy")

# print(a[:10])

# print(len(a))

# db_inf = np.zeros((105363, 2), dtype=int)

# i = 0
# for filename in tqdm.tqdm(os.listdir("./database/db")):
#         filepath = "./database/db/" + filename
#         data = np.load(filepath)
#         num = int(filename[:-4])
#         db_inf[i][0] = num
#         db_inf[i][1] = len(data)
#         i += 1

# np.save("./database/db_inf.npy", db_inf)


# a = torch.tensor([[1,2,3],[3,4,5]], dtype=torch.float32)
# b = torch.tensor([[3,4,5],[1,2,3]], dtype=torch.float32)

# c = F.cosine_similarity(a, b, dim = 1)
# print(c)


# center = np.zeros((len(os.listdir("./runs/retrieval/test/query")), 256))
# files = os.listdir("./runs/retrieval/test/query")
# for i in range(len(os.listdir("./runs/retrieval/test/query"))):
#         data = np.load("./runs/retrieval/test/query/"+ files[i])
#         if len(data) == 1:
#                 merge = np.zeros((1, 256))
#         else:
#                 merge = np.zeros((len(data)-1, 256))

#         for j in range(len(data)-1):
#                 merge[j] = np.concatenate((data[j], data[j+1]), axis=0)

#         if len(data) == 1:
#                 merge[0] = np.concatenate((data[0], data[0]), axis=0)

#         center[i] = np.mean(merge, axis=0)

# np.save("./runs/retrieval/test/query_mean_256.npy", center)

# a = np.load("./database/db_inf.npy")

# print(a[:10])


# center = np.zeros((len(os.listdir("./database/db")), 256))
# files = os.listdir("./database/db")
# for i in range(len(os.listdir("./database/db"))):
#         data = np.load("./database/db/"+ files[i])
#         if len(data) == 1:
#                 merge = np.zeros((1, 256))
#         else:
#                 merge = np.zeros((len(data)-1, 256))

#         for j in range(len(data)-1):
#                 merge[j] = np.concatenate((data[j], data[j+1]), axis=0)

#         if len(data) == 1:
#                 merge[0] = np.concatenate((data[0], data[0]), axis=0)

#         center[i] = np.mean(merge, axis=0)

# np.save("./database/db/db_mean_256.npy", center)




# sequence = np.random.random((5, 3))
# print(sequence)
# a = sequence[:2]
# b = sequence[2:]
# print(a)
# print(b)


# def read_csv(filename):
#     inf = dict()
#     with open(filename, 'r') as f:
#         reader = csv.reader(f)      
#         for i, row in enumerate(reader):
#             if i > 0:
#                 inf[row[1]] = row[0]

#     return inf

# inf = read_csv("./datasets/mt/val/val_song_inf.csv")
# # print(inf)

# for filename in os.listdir("./datasets/mt/train/data-0-1w"):
#     a = int(filename.split('-')[0])
#     b = int(filename.split('-')[1][:-4])
#     a = f"{a:05d}"

#     if b == 0:
#         new_name = f"{int(inf[a]):06d}.npy"
#         shutil.copy("./datasets/mt/val/data/"+filename, "./datasets/mt/train/data_old_name/"+new_name)



# a = np.random.rand(5,3)
# b = np.random.rand(1,3)

# print(a)
# print(b)
# c = np.vstack((a,b))
# print(c)


# df = pd.read_csv("./runs/checkpoint/mt_arcface/exp/result.csv")
# top1 = df['test/top1']
# print(top1)
# max_top1 = top1.max()
# print(max_top1)


# a = torch.load("./losses/cosine.pt")
# print(a.shape)

# nan_positions = []


# max_v, max_i = torch.max(a,dim=1)
# min_v, _ = torch.min(a,dim=1)

# print(max_v)
# # print(min_v)

# print(max_i)



# for i in range(a.size(0)):
#     print(i)
#     for j in range(a.size(1)):
#         if torch.isnan(a[i][j]):
#             nan_positions.append((i,j))


# print(nan_positions)



# def get_clean_embedding():
#     train_embedding = np.zeros((20000, 128))
#     model.eval()
#     with torch.no_grad():
#         for i in tqdm(range(1, 20001)):
#             data = np.load(f"./datasets/mt/train/data/{i}-00.npy")
#             sequence = np.zeros((100, 128))
#             sequence[:min(len(data), 100)] = data[:min(len(data), 100)]
#             sequence = np.float32(sequence)
#             sequence = torch.from_numpy(sequence).cuda()
#             sequence = sequence.unsqueeze(0)
#             local_embedding = model(sequence).squeeze(0).cpu().detach().numpy()
#             train_embedding[i - 1] = local_embedding
#     return train_embedding


# def update_arcface_center():
#     # center_a = np.load("./datasets/clean_embedder/train/clean_embedding.npy")
#     # center_b = np.load("./datasets/clean_embedder/val/clean_embedding.npy")
#     # center = np.vstack((center_a, center_b))
#     center = get_clean_embedding()
#     t_c = F.normalize(torch.from_numpy(center).float()).cuda()
#     arcface.weight = torch.nn.Parameter(t_c)




# for fi ,filename in enumerate(tqdm(os.listdir("./database/fma_full"))):
#     path = "./database/fma_full/" + filename
#     data = np.load(path)
#     for ri, i in enumerate(range(0, len(data), 29)):
#         chunk_len = min(29, len(data)-i)
#         if chunk_len != 29:
#             break
#         new_path = f"./database/fma_full_15s/{filename[:-4]}_{ri}.npy"
#         np.save(new_path, data[i:i+chunk_len])








# from losses.Tam import TamLoss

# checkpoint = torch.load("./runs/checkpoint/mt_test_arcface/exp/mt_test_1.pth")
# arcface = TamLoss(in_features=256, out_features=1000, s=64.0)
# arcface.load_state_dict(checkpoint["arcface"])
# print(arcface.weight[0])

# dic = dict()
# for filename in tqdm(os.listdir("G:\\code\\SMIR\\runs\\retrieval\\test\\db")):
#     path = "G:\\code\\SMIR\\runs\\retrieval\\test\\db/" + filename
#     a = np.load(path)
#     length = len(a)
#     if length in dic.keys():
#         dic[length]+=1
#     else:
#         dic[length]=0

# print(dic)


# for filename in tqdm(os.listdir("./datasets/mt/train/data-whole")):
#     num = int(filename.split('-')[0])
#     if num <= 1000:
#         old_path = "./datasets/mt/train/data-whole/" + filename
#         new_path = "./datasets/mt/train/data-1k-100/"+ filename
#         shutil.copy(old_path, new_path) 


# t = [1, 3, 5, 9, 11, 19, 39]
# d = dict()
# for item in t:
#     k = 59 - item
#     a = set()
#     while len(a) < 10:
#         id = random.randint(0, k)
#         a.add(id)

#     d[item] = list(a)


# print(d)

# with open("./runs/retrieval/test_ids.pickle", "wb") as file:
#     pickle.dump(d, file)


# # d = pickle.load(open("./runs/retrieval/test_ids.pickle", "rb"))
# # print(d)

# dummy_db_shape = np.load("./database/fma_full_30s_shape.npy")
# print(dummy_db_shape)


# import wave

# # 设置文件夹路径
# folder_path = "D:/neural-audio-fp-dataset/music/test-dummy-db-100k-full/train_data_10w"

# # 初始化计数器
# short_duration_count = 0
# is_not_ok = list()

# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, filename)
    
#     # 检查文件是否是.wav文件
#     if filename.endswith(".wav"):
#         with wave.open(file_path, "rb") as wav_file:
#             # 获取音频文件的帧数和帧率
#             num_frames = wav_file.getnframes()
#             frame_rate = wav_file.getframerate()
            
#             # 计算音频文件的时长（秒）
#             duration = num_frames / float(frame_rate)
            
#             # 检查时长是否小于25秒
#             if duration < 29.5:
#                 print(f"Short audio file: {filename}, Duration: {duration:.2f} seconds")
#                 is_not_ok.append(filename)
#                 short_duration_count += 1

# # 输出符合条件的文件数量
# print(f"Total short audio files found: {short_duration_count}")

# with open("./less_29.5.txt", "w") as file:
#     for item in is_not_ok:
#         file.write(item + "\n")


# load_list = []

# # with open("./less_29.5.txt", "r") as file:
# #     for line in file:
# #         load_list.append(line.strip())

# i = 1
# for filename in os.listdir(folder_path):
#     file_path = folder_path + "/" + filename
#     i+=1
#     if i > 100000:
#         os.remove(file_path)



# i = 0
# for filename in os.listdir("./runs/retrieval/test/tr-aug"):
#     path = "./runs/retrieval/test/tr-aug/" + filename
#     a = np.load(path)
#     name1 = filename[:-4] + '-1.npy'
#     name2 = filename[:-4] + '-2.npy'
#     path1 = "./runs/retrieval/test_15/tr-aug/" + name1
#     path2 = "./runs/retrieval/test_15/tr-aug/" + name2
#     a1 = a[:29]
#     a2 = a[29:]
#     np.save(path1, a1)
#     np.save(path2, a2)


# a1 = np.load("./runs/retrieval/test_15/db/000134-1.npy")
# a2 = np.load("./runs/retrieval/test_15/db/000134-2.npy")
# print(a1[-1])
# print(a2[0])


# test_ids_dict = pickle.load(open('./runs/retrieval/test_15/test_ids.pickle', 'rb'))
# print(test_ids_dict)

# print(1/5)

# k = [str(i) for i in list(test_ids_dict.keys())]
# print(k)


# a = 1234
# b = 5000
# print(a/b*100.)


# length = 12
# start_id = 5
# seq_len = 8
# s1 = np.zeros((32, 8))
# s2 = np.zeros((32, 8))
# data = np.random.random((32,8))
# for i in range(min(length, seq_len)):
#     s1[i] = data[i + start_id]
# s2[:min(length, seq_len)] = data[start_id: start_id + min(length, seq_len)]

# print(s1)
# print(s2)

# a = b = np.zeros((1,2))
# a[0,0]=1
# print(b)


a=b=0
print(a)
a=1
print(b)