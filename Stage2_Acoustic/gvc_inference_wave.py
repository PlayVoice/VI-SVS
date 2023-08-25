import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse

from omegaconf import OmegaConf
from scipy.io.wavfile import write
from bigvgan.model.generator import Generator
from pitch import load_csv_pitch


def load_bigv_model(checkpoint_path, model):
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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = OmegaConf.load(args.config)
    model = Generator(hp)
    load_bigv_model(args.model, model)
    model.eval()
    model.to(device)

    mel = torch.load(args.mel)

    pit = load_csv_pitch(args.pit)
    pit = torch.FloatTensor(pit)

    len_pit = pit.size()[0]
    len_mel = mel.size()[1]
    len_min = min(len_pit, len_mel)
    pit = pit[:len_min]
    mel = mel[:, :len_min]

    with torch.no_grad():
        mel = mel.unsqueeze(0).to(device)
        pit = pit.unsqueeze(0).to(device)
        audio = model.inference(mel, pit)
        audio = audio.cpu().detach().numpy()

        pitwav = model.pitch2wav(pit)
        pitwav = pitwav.cpu().detach().numpy()

    write("gvc_out.wav", hp.audio.sampling_rate, audio)
    write("gvc_pitch.wav", hp.audio.sampling_rate, pitwav)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mel', type=str,
                        help="Path of content vector.")
    parser.add_argument('--pit', type=str,
                        help="Path of pitch csv file.")
    args = parser.parse_args()

    args.config = "./bigvgan/configs/nsf_bigvgan.yaml"
    args.model = "./bigvgan_pretrain/nsf_bigvgan_pretrain_32K.pth"

    main(args)
