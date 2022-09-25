import os
import sys
import numpy as np

from scipy.io import wavfile
from time import *

import torch
import utils
from models import SynthesizerTrn


def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))


# define model and load checkpoint
hps = utils.get_hparams_from_file("./configs/singing_base.json")

net_g = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model,
).cuda()

_ = utils.load_checkpoint("./logs/singing_base/G_160000.pth", net_g, None)
net_g.eval()
# net_g.remove_weight_norm()

# check directory existence
if not os.path.exists("./singing_out"):
    os.makedirs("./singing_out")

idxs = [
    "2001000001",
    "2001000002",
    "2001000003",
    "2001000004",
    "2001000005",
    "2001000006",
    "2051001912",
    "2051001913",
    "2051001914",
    "2051001915",
    "2051001916",
    "2051001917",
]
for idx in idxs:
    phone = np.load(f"../VISinger_data/label_vits/{idx}_label.npy")
    score = np.load(f"../VISinger_data/label_vits/{idx}_score.npy")
    pitch = np.load(f"../VISinger_data/label_vits/{idx}_pitch.npy")
    slurs = np.load(f"../VISinger_data/label_vits/{idx}_slurs.npy")
    phone = torch.LongTensor(phone)
    score = torch.LongTensor(score)
    pitch = torch.LongTensor(pitch)
    slurs = torch.LongTensor(slurs)

    phone_lengths = phone.size()[0]

    begin_time = time()
    with torch.no_grad():
        phone = phone.cuda().unsqueeze(0)
        score = score.cuda().unsqueeze(0)
        pitch = pitch.cuda().unsqueeze(0)
        slurs = slurs.cuda().unsqueeze(0)
        phone_lengths = torch.LongTensor([phone_lengths]).cuda()
        audio = (
            net_g.infer(phone, phone_lengths, score, pitch, slurs)[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
    end_time = time()
    run_time = end_time - begin_time
    print("Syth Time (Seconds):", run_time)
    data_len = len(audio) / 16000
    print("Wave Time (Seconds):", data_len)
    print("Real time Rate (%):", run_time / data_len)
    save_wav(audio, f"./singing_out/singing_{idx}.wav", hps.data.sampling_rate)

# can be deleted
os.system("chmod 777 ./singing_out -R")
