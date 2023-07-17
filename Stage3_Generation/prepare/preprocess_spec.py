import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from spec.inference import mel_spectrogram_file
from tqdm import tqdm
from omegaconf import OmegaConf


def compute_spec(hps, filename, specname):
    spec = mel_spectrogram_file(filename, hps)
    spec = torch.squeeze(spec, 0)
    # print(spec.shape)
    torch.save(spec, specname)


def process_file(file):
    if file.endswith(".wav"):
        file = file[:-4]
        compute_spec(hps, f"{wavPath}/{spks}/{file}.wav", f"{spePath}/{spks}/{file}.mel.pt")

def process_files_with_thread_pool(wavPath, spks, max_workers):
    files = os.listdir(f"./{wavPath}/{spks}")
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        list(tqdm(executor.map(process_file, files), total=len(files)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-s", "--spe", help="spe", dest="spe")
    parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all cpu cores", dest="thread_count", type=int, default=1)
    args = parser.parse_args()
    print(args.wav)
    print(args.spe)

    os.makedirs(args.spe, exist_ok=True)
    wavPath = args.wav
    spePath = args.spe

    hps = OmegaConf.load(f"./configs/nsf_bigvgan.yaml")

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{spePath}/{spks}", exist_ok=True)
            if args.thread_count == 0:
                process_num = os.cpu_count()
            else:
                process_num = args.thread_count
            process_files_with_thread_pool(wavPath, spks, process_num)
