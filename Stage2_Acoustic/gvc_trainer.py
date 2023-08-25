import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import numpy as np

from omegaconf import OmegaConf
from grad_extend.train import train

torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./configs/base.yaml',
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    args = parser.parse_args()

    assert torch.cuda.is_available()
    print('Numbers of GPU :', torch.cuda.device_count())

    hps = OmegaConf.load(args.config)

    np.random.seed(hps.train.seed)
    torch.manual_seed(hps.train.seed)
    torch.cuda.manual_seed(hps.train.seed)

    train(hps, args.checkpoint_path)
