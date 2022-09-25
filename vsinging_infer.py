import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from time import *

import torch
import utils
from models import SynthesizerTrn
from prepare.data_vits import SingInput
from prepare.data_vits import FeatureInput


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

singInput = SingInput(16000, 256)
featureInput = FeatureInput("../VISinger_data/wav_dump_16k/", 16000, 256)

# check directory existence
if not os.path.exists("./singing_out"):
    os.makedirs("./singing_out")

fo = open("./vsinging_infer.txt", "r+")
while True:
    try:
        message = fo.readline().strip()
    except Exception as e:
        print("nothing of except:", e)
        break
    if message == None:
        break
    if message == "":
        break
    print(message)
    (
        file,
        labels_ids,
        labels_frames,
        scores_ids,
        scores_dur,
        labels_slr,
        labels_uvs,
    ) = singInput.parseInput(message)
    labels_ids = singInput.expandInput(labels_ids, labels_frames)
    labels_uvs = singInput.expandInput(labels_uvs, labels_frames)
    labels_slr = singInput.expandInput(labels_slr, labels_frames)
    scores_ids = singInput.expandInput(scores_ids, labels_frames)
    scores_pit = singInput.scorePitch(scores_ids)
    # elments by elments
    scores_pit_ = scores_pit * labels_uvs
    scores_pit = singInput.smoothPitch(scores_pit_)

    fig = plt.figure(figsize=(12, 6))
    plt.plot(scores_pit_.T, "g")
    plt.plot(scores_pit.T, "r")
    plt.savefig(f"./singing_out/{file}_f0_.png", format="png")
    plt.close(fig)

    phone = torch.LongTensor(labels_ids)
    score = torch.LongTensor(scores_ids)
    slurs = torch.LongTensor(labels_slr)
    pitch = featureInput.coarse_f0(scores_pit)
    pitch = torch.LongTensor(pitch)

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
    save_wav(audio, f"./singing_out/{file}.wav", hps.data.sampling_rate)
fo.close()
# can be deleted
os.system("chmod 777 ./singing_out -R")
