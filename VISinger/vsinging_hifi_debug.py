import os
import torch
import argparse
import numpy as np

import utils

from utils import load_wav_to_torch
from mel_processing import spectrogram_torch
from models import SynthesizerTrn

from scipy.io import wavfile


def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description='please enter embed parameter ...'
    parser.add_argument("-s", "--source", help="input wave", dest="source")

    args = parser.parse_args()
    source_file = args.source
    print("source file is :", source_file)

    hps = utils.get_hparams_from_file("./configs/singing_base.json")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()

    _ = utils.load_checkpoint("./logs/singing_base/G_50000.pth", net_g, None)
    _ = net_g.eval()

    audio, sampling_rate = load_wav_to_torch(source_file)

    # float
    y = audio / hps.data.max_wav_value
    y = y.unsqueeze(0)

    spec = spectrogram_torch(y, hps.data.filter_length,
        hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
        center=False).cuda()
    spec_lengths = torch.LongTensor([spec.size(-1)]).cuda()

    with torch.no_grad():
        audio_vc = net_g.voice_conversion(spec, spec_lengths)[0][0,0].data.cpu().float().numpy()

    os.system(f"cp {source_file} vc_in.wav")

    save_wav(audio_vc, 'vc_out.wav', hps.data.sampling_rate)
