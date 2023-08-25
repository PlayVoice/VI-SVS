import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import argparse
from omegaconf import OmegaConf
from grad.model import GradTTS


def load_model(checkpoint_path, model):
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


def main(args):
    hps = OmegaConf.load(args.config)

    print('Initializing Grad-TTS...')
    model = GradTTS(hps.grad.n_mels, hps.grad.n_vecs, hps.grad.n_pits, hps.grad.n_spks, hps.grad.n_embs,
                    hps.grad.n_enc_channels, hps.grad.filter_channels,
                    hps.grad.dec_dim, hps.grad.beta_min, hps.grad.beta_max, hps.grad.pe_scale)
    print('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))

    load_model(args.checkpoint_path, model)
    torch.save({'model': model.state_dict()}, "gvc.pth")
    torch.save({'model': model.state_dict()}, "gvc.pretrain.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./configs/base.yaml',
                        help="yaml file for config.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file for evaluation")
    args = parser.parse_args()

    main(args)
