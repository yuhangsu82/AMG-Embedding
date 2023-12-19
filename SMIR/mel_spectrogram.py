import os
import glob
import torchaudio
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from torch_audiomentations import (
    Compose,
    Gain,
    AddColoredNoise,
    ApplyImpulseResponse,
    AddBackgroundNoise,
    LowPassFilter,
    BandPassFilter,
    HighPassFilter,
    PitchShift,
)


# ir_paths = [
#     "G:/MyData/augmentation/ImpulseResponse/rir_1.wav",
#     "G:/MyData/augmentation/ImpulseResponse/rir_2.wav",
#     "G:/MyData/augmentation/ImpulseResponse/rir_3.wav",
#     "G:/MyData/augmentation/ImpulseResponse/rir_4.wav",
# ]

# bn_paths = ["G:/MyData/augmentation/BackgroundNoise/bn-1.wav"]

# apply_augmentation = Compose(
#     transforms=[
#         # Gain(
#         #     min_gain_in_db=-25.0,
#         #     max_gain_in_db=-24.0,
#         #     p=0.5,
#         # ),
#         # AddColoredNoise(min_snr_in_db=2.8, max_snr_in_db=3.0, min_f_decay=0.9, max_f_decay=1.0, p=0.5)
#         # AddColoredNoise(p=0.5),
#         ApplyImpulseResponse(ir_paths=ir_paths, p=1.0)
#         # AddBackgroundNoise(background_paths=bn_paths, p=0.5)
#         # HighPassFilter(p=0.5),
#         # LowPassFilter(p=0.5),
#         # PitchShift(
#         #     min_transpose_semitones=-1.0,
#         #     max_transpose_semitones=1.0,
#         #     sample_rate=16000,
#         #     p=1.0,
#         # )
#         # HighPassFilter(p=1.0)
#     ]
# )

# apply_augmentation = Compose(
#     transforms=[
#         # Gain(
#         #     min_gain_in_db=-25.0,
#         #     max_gain_in_db=-24.0,
#         #     p=0.5,
#         # ),
#         # AddColoredNoise(min_snr_in_db=2.5, p=0.5),
#         # ApplyImpulseResponse(ir_paths=ir_paths, p=0.5),
#         # AddBackgroundNoise(background_paths=bn_paths, p=0.5),
#         HighPassFilter(min_cutoff_freq=550, max_cutoff_freq=600, p=1.0),
#         # LowPassFilter(min_cutoff_freq=2400, p=0.5),
#     ]
# )

ir_paths = [path for path in glob.glob("./datasets/augmentation/ImpulseResponse/*")]
bn_paths = [path for path in glob.glob("./datasets/augmentation/BackgroundNoise/*")]

apply_augmentation = Compose(
    transforms=[
        Gain(
            min_gain_in_db=-25.0,
            max_gain_in_db=-20.0,
            p=0.5,
        ),
        AddColoredNoise(min_snr_in_db=2.5, p=0.5),
        ApplyImpulseResponse(ir_paths=ir_paths, p=0.5),
        AddBackgroundNoise(background_paths=bn_paths, p=0.5),
        HighPassFilter(p=0.5),
        LowPassFilter(min_cutoff_freq=2400, p=0.5),
        # PitchShift(
        #     min_transpose_semitones=-1.0,
        #     max_transpose_semitones=1.0,
        #     p=0.2,
        #     sample_rate=16000,
        # ),
    ]
)

index = 1

# # 样本
# audio_path = f"G:/code/python/SMIR/datasets/val/wav_raw/val_raw_chunk{index}.wav"
# img_path_1 = f"G:/code/python/SMIR/test/photo/val_raw_chunk{index}.jpg"


# waveform, sr = torchaudio.load(audio_path)
# # waveform = apply_augmentation(waveform.unsqueeze(0), sample_rate=16000).squeeze(0)
# mel_spectrogram = torchaudio.transforms.MelSpectrogram(
#     sample_rate=16000, n_fft=1024, win_length=1024, hop_length=512, n_mels=128
# )(waveform)

# resize = transforms.Resize(
#     [313, 313], interpolation=transforms.InterpolationMode.BILINGAR
# )
# db_spectrogram = resize(
#     torchaudio.transforms.AmplitudeToDB()(mel_spectrogram).unsqueeze(0)
# )

# db_spectrogram = db_spectrogram.squeeze(0).squeeze(0)

# spectrogram = (
#     (
#         (db_spectrogram - db_spectrogram.min())
#         / (db_spectrogram.max() - db_spectrogram.min())
#         * 255
#     )
#     .numpy()
#     .astype(np.uint8)
# )

# dst = cv2.applyColorMap(spectrogram, cv2.COLORMAP_MAGMA)

# cv2.imwrite(img_path_1, dst)


# # 录音
# audio_path = f"G:/code/python/SMIR/datasets/val/wav_record/val_record_chunk{index}.wav"
# img_path_2 = f"G:/code/python/SMIR/test/photo/val_record_chunk{index}.jpg"

# waveform, sr = torchaudio.load(audio_path)
# # waveform = apply_augmentation(waveform.unsqueeze(0), sample_rate=16000).squeeze(0)
# mel_spectrogram = torchaudio.transforms.MelSpectrogram(
#     sample_rate=16000, n_fft=1024, win_length=1024, hop_length=512, n_mels=128
# )(waveform)

# resize = transforms.Resize(
#     [313, 313], interpolation=transforms.InterpolationMode.BILINGAR
# )
# db_spectrogram = resize(
#     torchaudio.transforms.AmplitudeToDB()(mel_spectrogram).unsqueeze(0)
# )

# db_spectrogram = db_spectrogram.squeeze(0).squeeze(0)

# spectrogram = (
#     (
#         (db_spectrogram - db_spectrogram.min())
#         / (db_spectrogram.max() - db_spectrogram.min())
#         * 255
#     )
#     .numpy()
#     .astype(np.uint8)
# )

# dst = cv2.applyColorMap(spectrogram, cv2.COLORMAP_MAGMA)

# cv2.imwrite(img_path_2, dst)


# # 数据增强
# audio_path = f"G:/code/python/SMIR/datasets/val/wav_raw/val_raw_chunk{index}.wav"
# img_path_3 = f"G:/code/python/SMIR/test/photo/val_raw_chunk{index}_aug.jpg"

# waveform, sr = torchaudio.load(audio_path)
# waveform = apply_augmentation(waveform.unsqueeze(0), sample_rate=16000).squeeze(0)
# mel_spectrogram = torchaudio.transforms.MelSpectrogram(
#     sample_rate=16000, n_fft=1024, win_length=1024, hop_length=512, n_mels=128
# )(waveform)

# resize = transforms.Resize(
#     [313, 313], interpolation=transforms.InterpolationMode.BILINGAR
# )
# db_spectrogram = resize(
#     torchaudio.transforms.AmplitudeToDB()(mel_spectrogram).unsqueeze(0)
# )

# db_spectrogram = db_spectrogram.squeeze(0).squeeze(0)

# spectrogram = (
#     (
#         (db_spectrogram - db_spectrogram.min())
#         / (db_spectrogram.max() - db_spectrogram.min())
#         * 255
#     )
#     .numpy()
#     .astype(np.uint8)
# )

# dst = cv2.applyColorMap(spectrogram, cv2.COLORMAP_MAGMA)

# cv2.imwrite(img_path_3, dst)


# # 裁剪+高能区域
# audio_path = f"G:/code/python/SMIR/datasets/val/wav_record/val_record_chunk{index}.wav"
# img_path_4 = f"G:/code/python/SMIR/test/photo/val_record_chunk{index}_zero.jpg"

# waveform, sr = torchaudio.load(audio_path)
# mel_spectrogram = torchaudio.transforms.MelSpectrogram(
#     sample_rate=16000, n_fft=1024, win_length=1024, hop_length=512, n_mels=128
# )(waveform)

# resize = transforms.Resize(
#     [313, 313], interpolation=transforms.InterpolationMode.BILINGAR
# )
# db_spectrogram = resize(
#     torchaudio.transforms.AmplitudeToDB()(mel_spectrogram).unsqueeze(0)
# )

# # crop = transforms.RandomCrop((248, 313), pad_if_needed=False, padding_mode='constant', fill=0)
# # crop.topleft = (40, 0)
# # db_spectrogram = crop(db_spectrogram).squeeze(0).squeeze(0)

# db_spectrogram = db_spectrogram[:, :, 50:288, :].squeeze(0).squeeze(0)
# print(db_spectrogram.shape)

# spectrogram = (
#     (db_spectrogram - db_spectrogram.min())
#     / (db_spectrogram.max() - db_spectrogram.min())
#     * 255
# )

# spectrogram[spectrogram < 0.6 * spectrogram.mean()] = 0
# spectrogram = spectrogram.numpy().astype(np.uint8)

# dst = cv2.applyColorMap(spectrogram, cv2.COLORMAP_MAGMA)

# cv2.imwrite(img_path_4, dst)


# # 显示对比图
# img1 = cv2.imread(img_path_1)
# img2 = cv2.imread(img_path_2)
# img3 = cv2.imread(img_path_4)

# # 新建一个画布，并设置为横向排列
# fig, axis = plt.subplots(1, 3)

# # 在画布上显示每张图片
# axis[0].imshow(img1[..., ::-1])  # opencv读取的图片是BGR格式，需要转换为RGB格式
# axis[0].set_title("image1")
# axis[1].imshow(img2[..., ::-1])
# axis[1].set_title("image2")
# axis[2].imshow(img3[..., ::-1])
# axis[2].set_title("image3")

# # 显示画布
# plt.show()


# for i in range(20):
#     audio_path = f"./runs/retrieval/test-30s/000001.wav"
#     img_path_3 = f"./test/photo/000001_aug{i}.jpg"

#     waveform, sr = torchaudio.load(audio_path)
#     waveform = apply_augmentation(waveform.unsqueeze(0), sample_rate=16000).squeeze(0)
#     mel_spectrogram = torchaudio.transforms.MelSpectrogram(
#         sample_rate=16000, n_fft=1024, win_length=1024, hop_length=512, n_mels=128
#     )(waveform)

#     resize = transforms.Resize(
#         [313, 313], interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
#     )

#     db_spectrogram = resize(
#         torchaudio.transforms.AmplitudeToDB()(mel_spectrogram).unsqueeze(0)
#     )

#     db_spectrogram = db_spectrogram.squeeze(0).squeeze(0)

#     spectrogram = (
#         (db_spectrogram - db_spectrogram.min())
#         / (db_spectrogram.max() - db_spectrogram.min())
#         * 255
#     )
#     spectrogram[spectrogram < 0.6 * spectrogram.mean()] = 0
#     spectrogram = spectrogram.numpy().astype(np.uint8)

#     dst = cv2.applyColorMap(spectrogram, cv2.COLORMAP_MAGMA)

#     cv2.imwrite(img_path_3, dst)


def draw_mel(wav_path, output_path, aug=False):

    # print(wav_path)
    waveform, sr = torchaudio.load(wav_path, normalize=True)
    length = (len(waveform[0]) // sr) * sr
    slices = range(0, length - 2 * sr + 1, sr)

    for j, start in enumerate(slices):
        end = slices[j] + 2 * sr
        chunk = waveform[0][start:end].unsqueeze(0)
        if aug == True:
            chunk = apply_augmentation(chunk.unsqueeze(0), sample_rate=16000).squeeze(0)

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, win_length=1024, hop_length=256, n_mels=128, f_min=300, f_max=4000
        )(chunk)

        # resize = transforms.Resize(
        #     [128, 128],
        #     interpolation=transforms.InterpolationMode.BILINEAR,
        #     antialias=True,
        # )
        # mel_spec = resize(
        #     torchaudio.transforms.AmplitudeToDB()(mel_spec).unsqueeze(0)
        # )
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec).unsqueeze(0)
        db_spec = mel_spec.squeeze(0).squeeze(0)

        mel_spec = (db_spec - db_spec.min()) / (db_spec.max() - db_spec.min()) * 255
        k = random.uniform(0.0, 1.5)
        mel_spec[mel_spec < k * mel_spec.mean()] = 0
        mel_spec = mel_spec.numpy().astype(np.uint8)
        img = cv2.applyColorMap(mel_spec, cv2.COLORMAP_MAGMA)

        file_path = output_path + f"/chunk-{j}.jpg"

        cv2.imwrite(file_path, img)

# draw_mel("F:/split-wav-in/song_1.wav","G:/code/SMIR/test/img/test")
# draw_mel("F:/MyData/new-dataset/song-16k/blues.00000.wav","G:/code/SMIR/test/img/test1")
# draw_mel("F:/MyData/new-dataset/song-16k/blues.00000.wav","G:/code/SMIR/test/img/aug", aug=True)


# audio_path = f"./runs/retrieval/test-30s/000001.wav"
# img_path_3 = f"./test/photo/000001_aug1.wav"



# waveform, sr = torchaudio.load(audio_path)
# waveform = apply_augmentation(waveform.unsqueeze(0), sample_rate=16000).squeeze(0)
# torchaudio.save(img_path_3, waveform, sample_rate=16000)




# # 将一个wav音频剪成多个2s的片段
# waveform, sr = torchaudio.load("./runs/retrieval/test-30s/000072.wav")
# length = (len(waveform[0]) // sr) * sr
# slices = range(0, length - 2 * sr + 1, sr)
# print(len(slices))
# for j, start in enumerate(slices):
#     end = slices[j] + 2 * sr
#     chunk = waveform[0][start:end].unsqueeze(0)
#     torchaudio.save(f"./test/chunk-test/chunk-{j}.wav", chunk, sample_rate=16000)


for filename in os.listdir("./runs/retrieval/audio1"):
    waveform, sr = torchaudio.load("./runs/retrieval/audio1/" + filename)
    waveform = apply_augmentation(waveform.unsqueeze(0), sample_rate=16000).squeeze(0)
    torchaudio.save("./runs/retrieval/audio1_aug/" + filename, waveform, sample_rate=16000)