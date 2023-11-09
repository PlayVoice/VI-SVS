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

from pitch.models import PitchDiffusion
from pitch.utils import fix_len_compatibility


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
    parser.add_argument('-p', '--pitch', type=str, required=True,
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

    net_p = PitchDiffusion().cuda()
    net_p.eval()
    load_svs_model(args.pitch, net_p)

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

        phone = torch.LongTensor(labels_ids)
        score = torch.LongTensor(scores_ids)
        slurs = torch.LongTensor(labels_slr)

        lengths = phone.size()[0]
        lengths_fix = fix_len_compatibility(lengths)

        phone_fix = torch.zeros((1, lengths_fix), dtype=torch.long).cuda()
        score_fix = torch.zeros((1, lengths_fix), dtype=torch.long).cuda()
        slurs_fix = torch.zeros((1, lengths_fix), dtype=torch.long).cuda()
        phone_fix[0, :lengths] = phone
        score_fix[0, :lengths] = score
        slurs_fix[0, :lengths] = slurs

        with torch.no_grad():
            n_timesteps = 50
            temperature = 1
            # PIT
            phone_lengths = torch.LongTensor([lengths_fix]).cuda()
            pit_pri, pit_pre = net_p(phone_fix, phone_lengths, score_fix, slurs_fix, n_timesteps, temperature)
            pitch = pit_pre[:, 0, :]
            pitch = 2**pitch
            print('~~~~~~~')
            # SVS
            audio = (
                net_g.infer(phone_fix, phone_lengths, score_fix, pitch, slurs_fix)[0, 0]
                .data.cpu()
                .float()
                .numpy()
            )

        save_wav(audio, f"./svs_out/{file}.wav", hps.data.sampling_rate)
    fo.close()
    # can be deleted
    os.system("chmod 777 ./svs_out -R")
