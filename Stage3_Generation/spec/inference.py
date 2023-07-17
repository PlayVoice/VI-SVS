# Adapted from https://github.com/jik876/hifi-gan under the MIT license.

import argparse
import torch
import torch.utils.data
import numpy as np
import librosa
from omegaconf import OmegaConf
from librosa.filters import mel as librosa_mel_fn


MAX_WAV_VALUE = 32768.0


def load_wav_to_torch(full_path, sample_rate):
    wav, _ = librosa.load(full_path, sr=sample_rate)
    wav = wav / np.abs(wav).max() * 0.6
    return torch.FloatTensor(wav)


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def mel_spectrogram_file(path, hps):
    audio = load_wav_to_torch(path, hps.audio.sampling_rate)
    audio = audio.unsqueeze(0)

    # match audio length to self.hop_length * n for evaluation
    if (audio.size(1) % hps.audio.hop_length) != 0:
        audio = audio[:, :-(audio.size(1) % hps.audio.hop_length)]
    mel = mel_spectrogram(audio, hps.audio.filter_length, hps.audio.n_mel_channels, hps.audio.sampling_rate,
                          hps.audio.hop_length, hps.audio.win_length, hps.audio.mel_fmin, hps.audio.mel_fmax, center=False)
    return mel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-m", "--mel", help="mel", dest="mel")  # csv for excel
    args = parser.parse_args()
    print(args.wav)
    print(args.mel)

    hps = OmegaConf.load(f"./configs/nsf_bigvgan.yaml")

    mel = mel_spectrogram_file(args.wav, hps)
    # TODO
    mel = torch.squeeze(mel, 0)
    # [100, length]
    torch.save(mel, args.mel)
