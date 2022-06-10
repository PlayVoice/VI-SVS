import os, sys
import librosa
import numpy as np

from scipy.io import wavfile


def load_wave(wav_fpath):
    wavdata = []
    wav, _ = librosa.load(wav_fpath, 16000)
    wav = wav / np.abs(wav).max() * 0.6
    wavdata.extend(wav)
    wavdata = np.array(wavdata, dtype="float32")
    return wavdata


def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, 16000, wav.astype(np.int16))


rootPath = "VISinger_data/wavs"
outPath = "VISinger_data/wav_dump_16k"

for spks in os.listdir(rootPath):
    if os.path.isdir(f"./{rootPath}/{spks}"):
        os.makedirs(f"./{outPath}/{spks}")
        for file in os.listdir(f"./{rootPath}/{spks}"):
            if file.endswith(".wav"):
                tmp = load_wave(f"./{rootPath}/{spks}/{file}")
                save_wav(tmp, f"./{outPath}/{spks}/{file}")
    elif spks.endswith(".wav"):
        file = spks
        tmp = load_wave(f"./{rootPath}/{file}")
        save_wav(tmp, f"./{outPath}/{file}")
