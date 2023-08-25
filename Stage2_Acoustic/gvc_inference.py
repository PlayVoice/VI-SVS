import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import argparse
import numpy as np

from omegaconf import OmegaConf
from pitch import load_csv_pitch
from spec.inference import print_mel

from grad.utils import fix_len_compatibility
from grad.model import GradTTS


def load_gvc_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    saved_state_dict = checkpoint_dict["model"]
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


@torch.no_grad()
def gvc_main(device, model, _vec, _pit, spk):
    l_vec = _vec.shape[0]
    d_vec = _vec.shape[1]
    lengths_fix = fix_len_compatibility(l_vec)
    lengths = torch.LongTensor([l_vec]).to(device)
    vec = torch.zeros((1, lengths_fix, d_vec), dtype=torch.float32).to(device)
    pit = torch.zeros((1, lengths_fix), dtype=torch.float32).to(device)
    vec[0, :l_vec, :] = _vec
    pit[0, :l_vec] = _pit
    y_enc, y_dec = model(lengths, vec, pit, spk, n_timesteps=50)
    y_dec = y_dec.squeeze(0)
    y_dec = y_dec[:, :l_vec]
    return y_dec


def main(args):

    if (args.vec == None):
        args.vec = "gvc_tmp.vec.npy"
        print(
            f"Auto run : python hubert/inference.py -w {args.wave} -v {args.vec}")
        os.system(f"python hubert/inference.py -w {args.wave} -v {args.vec}")

    if (args.pit == None):
        args.pit = "gvc_tmp.pit.csv"
        print(
            f"Auto run : python pitch/inference.py -w {args.wave} -p {args.pit}")
        os.system(f"python pitch/inference.py -w {args.wave} -p {args.pit}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hps = OmegaConf.load(args.config)

    print('Initializing Grad-TTS...')
    model = GradTTS(hps.grad.n_mels, hps.grad.n_vecs, hps.grad.n_pits, hps.grad.n_spks, hps.grad.n_embs,
                    hps.grad.n_enc_channels, hps.grad.filter_channels,
                    hps.grad.dec_dim, hps.grad.beta_min, hps.grad.beta_max, hps.grad.pe_scale)
    print('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))

    load_gvc_model(args.model, model)
    model.eval()
    model.to(device)

    spk = np.load(args.spk)
    spk = torch.FloatTensor(spk)

    vec = np.load(args.vec)
    vec = np.repeat(vec, 2, 0)
    vec = torch.FloatTensor(vec)

    pit = load_csv_pitch(args.pit)
    pit = np.array(pit)
    pit = pit * 2 ** (args.shift / 12)
    pit = torch.FloatTensor(pit)

    len_pit = pit.size()[0]
    len_vec = vec.size()[0]
    len_min = min(len_pit, len_vec)
    pit = pit[:len_min]
    vec = vec[:len_min, :]

    with torch.no_grad():
        spk = spk.unsqueeze(0).to(device)

        all_frame = len_min
        hop_frame = 8
        out_chunk = 2400  # 24 S
        out_index = 0
        out_mel = None

        while (out_index < all_frame):
            if (out_index == 0):  # start frame
                cut_s = 0
                cut_s_out = 0
            else:
                cut_s = out_index - hop_frame
                cut_s_out = hop_frame

            if (out_index + out_chunk + hop_frame > all_frame):  # end frame
                cut_e = all_frame
                cut_e_out = -1
            else:
                cut_e = out_index + out_chunk + hop_frame
                cut_e_out = -1 * hop_frame

            sub_vec = vec[cut_s:cut_e, :].to(device)
            sub_pit = pit[cut_s:cut_e].to(device)

            sub_out = gvc_main(device, model, sub_vec, sub_pit, spk)
            sub_out = sub_out[:, cut_s_out:cut_e_out]
 
            out_index = out_index + out_chunk
            if out_mel == None:
                out_mel = sub_out
            else:
                out_mel = torch.cat((out_mel, sub_out), -1)
            if cut_e == all_frame:
                break

    torch.save(out_mel, "gvc_out.mel.pt")
    print_mel(out_mel, "gvc_out.mel.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/base.yaml',
                        help="yaml file for config.")
    parser.add_argument('--model', type=str, required=True,
                        help="path of model for evaluation")
    parser.add_argument('--wave', type=str, required=True,
                        help="Path of raw audio.")
    parser.add_argument('--spk', type=str, required=True,
                        help="Path of speaker.")
    parser.add_argument('--vec', type=str,
                        help="Path of hubert vector.")
    parser.add_argument('--pit', type=str,
                        help="Path of pitch csv file.")
    parser.add_argument('--shift', type=int, default=0,
                        help="Pitch shift key.")
    args = parser.parse_args()

    main(args)
