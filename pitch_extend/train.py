import os
import time
import logging
import math
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from vits.commons import clip_grad_value_

from pitch.utils import fix_len_compatibility
from pitch.models import PitchDiffusion
from pitch_extend.validation import validate
from pitch_extend.writer import MyWriter
from pitch_extend.dataloader import create_dataloader_train
from pitch_extend.dataloader import create_dataloader_eval


def load_model(model, saved_state_dict):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    return model


# 400 frames
out_size = fix_len_compatibility(400)


def train(rank, args, chkpt_path, hp, hp_str):

    if args.num_gpus > 1:
        init_process_group(backend=hp.dist_config.dist_backend, init_method=hp.dist_config.dist_url,
                           world_size=hp.dist_config.world_size * args.num_gpus, rank=rank)

    torch.cuda.manual_seed(hp.train.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    model_g = PitchDiffusion().to(device)

    optim_g = torch.optim.AdamW(model_g.parameters(),
                                lr=hp.train.learning_rate, betas=hp.train.betas, eps=hp.train.eps)

    init_epoch = 1
    step = 0

    # define logger, writer, valloader, stft at rank_zero
    if rank == 0:
        pth_dir = os.path.join(hp.log.pth_dir, args.name)
        log_dir = os.path.join(hp.log.log_dir, args.name)
        os.makedirs(pth_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (args.name, time.time()))),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger()
        writer = MyWriter(hp, log_dir)
        valloader = create_dataloader_eval(hp)

    if chkpt_path is not None:
        if rank == 0:
            logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path, map_location='cpu')
        load_model(model_g, checkpoint['model_g'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        init_epoch = checkpoint['epoch']
        step = checkpoint['step']

        if rank == 0:
            if hp_str != checkpoint['hp_str']:
                logger.warning("New hparams is different from checkpoint. Will use new.")
    else:
        if rank == 0:
            logger.info("Starting new training run.")

    if args.num_gpus > 1:
        model_g = DistributedDataParallel(model_g, device_ids=[rank])

    # this accelerates training when the size of minibatch is always consistent.
    # if not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hp.train.lr_decay, last_epoch=init_epoch-2)
    trainloader = create_dataloader_train(hp, args.num_gpus, rank)

    for epoch in range(init_epoch, hp.train.epochs):

        trainloader.batch_sampler.set_epoch(epoch)

        if rank == 0 and epoch % hp.log.eval_interval == 0:
            with torch.no_grad():
                validate(hp, model_g, valloader, writer, step, device)

        if rank == 0:
            loader = tqdm.tqdm(trainloader, desc='Loading train data')
        else:
            loader = trainloader

        model_g.train()

        for phone, phone_l, score, pitch, slurs in loader:

            phone = phone.to(device)
            phone_l = phone_l.to(device)
            score = score.to(device)
            pitch = pitch.to(device)
            slurs = slurs.to(device)

            # generator
            optim_g.zero_grad()
            #
            prior_loss, diff_loss = model_g.compute_loss(phone, phone_l, score, slurs, pitch, out_size=out_size)
            loss_g = sum([prior_loss, diff_loss])
            loss_g.backward()
            clip_grad_value_(model_g.parameters(),  None)
            optim_g.step()

            step += 1
            # logging
            loss_g = loss_g.item()
            if rank == 0 and step % hp.log.info_interval == 0:
                writer.log_training(loss_g, prior_loss, diff_loss, step)
                logger.info("epoch %d | g %.04f prior_loss %.04f diff_loss %.04f | step %d" % (
                    epoch, loss_g, prior_loss, diff_loss, step))

        if rank == 0 and epoch % hp.log.save_interval == 0:
            save_path = os.path.join(pth_dir, '%s_%04d.pt'
                                     % (args.name, epoch))
            torch.save({
                'model_g': (model_g.module if args.num_gpus > 1 else model_g).state_dict(),
                'optim_g': optim_g.state_dict(),
                'step': step,
                'epoch': epoch,
                'hp_str': hp_str,
            }, save_path)
            logger.info("Saved checkpoint to: %s" % save_path)

        scheduler_g.step()
