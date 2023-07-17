import os
import torch
import random
import numpy as np
from scipy.io.wavfile import read
from torch.utils.data import DataLoader, Dataset


def read_wav_np(path):
    sr, wav = read(path)
    if len(wav.shape) == 2:
        wav = wav[:, 0]
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0
    wav = wav.astype(np.float32)
    return sr, wav


def load_items(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths = []
        for line in f:
            path_text = line.strip().split(split)
            filepaths.append(path_text)
    return filepaths


def create_dataloader(hp, train):
    dataset = FeatureFromDisk(hp, train)
    if train:
        return DataLoader(dataset=dataset, batch_size=hp.train.batch_size, shuffle=True,
                          num_workers=hp.train.num_workers, pin_memory=True, drop_last=True)
    else:
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                          num_workers=hp.train.num_workers, pin_memory=True, drop_last=False)


class FeatureFromDisk(Dataset):
    def __init__(self, hp, train):
        self.hp = hp
        self.frame_segment_length = hp.audio.segment_length // hp.audio.hop_length
        self.train = train
        if self.train:
            path = hp.data.train_file
        else:
            path = hp.data.val_file
        self.items = load_items(path)
        self._filter()
        print(f'----------{len(self.items)}----------')

    def _filter(self):
        items_new = []
        for wavpath, pitch, mel in self.items:
            if not os.path.isfile(wavpath):
                continue
            if not os.path.isfile(pitch):
                continue
            if not os.path.isfile(mel):
                continue
            sr, audio = read_wav_np(wavpath)
            assert sr == self.hp.audio.sampling_rate
            if len(audio) > self.hp.audio.segment_length * 2:
                items_new.append([wavpath, pitch, mel])
        self.items = items_new

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.my_getitem(idx)

    def my_getitem(self, idx):
        item = self.items[idx]
        wav = item[0]
        pit = item[1]
        mel = item[2]

        sr, wav = read_wav_np(wav)
        wav = torch.from_numpy(wav).unsqueeze(0)
        pit = np.load(pit)
        mel = torch.load(mel)

        pit = torch.FloatTensor(pit)
        mel = torch.FloatTensor(mel)

        len_pit = pit.size()[0]
        len_mel = mel.size()[1]
        len_min = min(len_pit, len_mel)
        len_wav = len_min * self.hp.audio.hop_length

        pit = pit[:len_min]
        mel = mel[:, :len_min]
        wav = wav[:, :len_wav]
        if self.train:
            max_frame_start = mel.size(1) - self.frame_segment_length - 1
            frame_start = random.randint(0, max_frame_start)
            frame_end = frame_start + self.frame_segment_length
            mel = mel[:, frame_start:frame_end]
            pit = pit[frame_start:frame_end]

            wav_start = frame_start * self.hp.audio.hop_length
            wav_len = self.hp.audio.segment_length
            wav = wav[:, wav_start:wav_start + wav_len]
        return mel, pit, wav
