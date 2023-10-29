import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from time import *

import torch
import argparse

from vits.models import SynthesizerTrn
from util import SingInput
from util import FeatureInput
from omegaconf import OmegaConf


def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))


def load_svs_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    saved_state_dict = checkpoint_dict["model_g"]
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="path of checkpoint pt file")
    args = parser.parse_args()

    # define model and load checkpoint
    hps = OmegaConf.load(args.config)

    singInput = SingInput(hps.data.sampling_rate, hps.data.hop_length)
    featureInput = FeatureInput(hps.data.sampling_rate, hps.data.hop_length)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.data.segment_size // hps.data.hop_length,
        hps).cuda()
    net_g.eval()

    load_svs_model(args.model, net_g)

    # check directory existence
    os.makedirs("./svs_out", exist_ok=True)
    fo = open("./svs_infer.txt", "r+")
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
        plt.savefig(f"./svs_out/{file}_f0_.png", format="png")
        plt.close(fig)

        phone = torch.LongTensor(labels_ids)
        score = torch.LongTensor(scores_ids)
        slurs = torch.LongTensor(labels_slr)
        pitch = torch.FloatTensor(scores_pit)

        phone_lengths = phone.size()[0]

        with torch.no_grad():
            phone = phone.cuda().unsqueeze(0)
            score = score.cuda().unsqueeze(0)
            pitch = pitch.cuda().unsqueeze(0)
            slurs = slurs.cuda().unsqueeze(0)
            phone_lengths = torch.LongTensor([phone_lengths]).cuda()
            audio = (
                net_g.infer(phone, phone_lengths, score, pitch, slurs)[0, 0]
                .data.cpu()
                .float()
                .numpy()
            )

        save_wav(audio, f"./svs_out/{file}.wav", hps.data.sampling_rate)
    fo.close()
    # can be deleted
    os.system("chmod 777 ./svs_out -R")
