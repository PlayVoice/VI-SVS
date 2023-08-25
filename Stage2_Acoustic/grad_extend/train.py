import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from grad_extend.data import TextMelSpeakerDataset, TextMelSpeakerBatchCollate
from grad_extend.utils import plot_tensor, save_plot, load_model
from grad.utils import fix_len_compatibility
from grad.model import GradTTS


# 200 frames
out_size = fix_len_compatibility(200)


def train(hps, chkpt_path=None):

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=hps.train.log_dir)

    print('Initializing data loaders...')
    train_dataset = TextMelSpeakerDataset(hps.train.train_files)
    batch_collate = TextMelSpeakerBatchCollate()
    loader = DataLoader(dataset=train_dataset,
                        batch_size=hps.train.batch_size,
                        collate_fn=batch_collate,
                        drop_last=True,
                        num_workers=8,
                        shuffle=True)
    test_dataset = TextMelSpeakerDataset(hps.train.valid_files)

    print('Initializing model...')
    model = GradTTS(hps.grad.n_mels, hps.grad.n_vecs, hps.grad.n_pits, hps.grad.n_spks, hps.grad.n_embs,
                    hps.grad.n_enc_channels, hps.grad.filter_channels,
                    hps.grad.dec_dim, hps.grad.beta_min, hps.grad.beta_max, hps.grad.pe_scale).cuda()
    print('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))

    # Load Pretrain
    if os.path.isfile(hps.train.pretrain):
        print("Start from Grad_SVC pretrain model: %s" % hps.train.pretrain)
        checkpoint = torch.load(hps.train.pretrain, map_location='cpu')
        load_model(model, checkpoint['model'])
        hps.train.learning_rate = 2e-5

    print('Initializing optimizer...')
    optim = torch.optim.Adam(params=model.parameters(), lr=hps.train.learning_rate)

    initepoch = 1
    iteration = 0

    # Load Continue
    if chkpt_path is not None:
        print("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optim'])
        initepoch = checkpoint['epoch']
        iteration = checkpoint['steps']

    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=hps.train.test_size)
    for i, item in enumerate(test_batch):
        mel = item['mel']
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{hps.train.log_dir}/original_{i}.png')

    print('Start training...')
    skip_diff_train = True
    for epoch in range(initepoch, hps.train.n_epochs + 1):
        model.eval()
        print('Synthesis...')

        if epoch % hps.train.test_step == 0:
            with torch.no_grad():
                for i, item in enumerate(test_batch):
                    l_vec = item['vec'].shape[0]
                    d_vec = item['vec'].shape[1]

                    lengths_fix = fix_len_compatibility(l_vec)
                    lengths = torch.LongTensor([l_vec]).cuda()

                    vec = torch.zeros((1, lengths_fix, d_vec), dtype=torch.float32).cuda()
                    pit = torch.zeros((1, lengths_fix), dtype=torch.float32).cuda()
                    spk = item['spk'].to(torch.float32).unsqueeze(0).cuda()
                    vec[0, :l_vec, :] = item['vec']
                    pit[0, :l_vec] = item['pit']

                    y_enc, y_dec = model(lengths, vec, pit, spk, n_timesteps=50)

                    logger.add_image(f'image_{i}/generated_enc',
                                     plot_tensor(y_enc.squeeze().cpu()),
                                     global_step=iteration, dataformats='HWC')
                    logger.add_image(f'image_{i}/generated_dec',
                                     plot_tensor(y_dec.squeeze().cpu()),
                                     global_step=iteration, dataformats='HWC')
                    save_plot(y_enc.squeeze().cpu(), 
                              f'{hps.train.log_dir}/generated_enc_{i}.png')
                    save_plot(y_dec.squeeze().cpu(), 
                              f'{hps.train.log_dir}/generated_dec_{i}.png')

        model.train()

        prior_losses = []
        diff_losses = []
        mel_losses = []
        spk_losses = []
        with tqdm(loader, total=len(train_dataset)//hps.train.batch_size) as progress_bar:
            for batch in progress_bar:
                model.zero_grad()

                lengths = batch['lengths'].cuda()
                vec = batch['vec'].cuda()
                pit = batch['pit'].cuda()
                spk = batch['spk'].cuda()
                mel = batch['mel'].cuda()

                prior_loss, diff_loss, mel_loss, spk_loss = model.compute_loss(
                    lengths, vec, pit, spk,
                    mel, out_size=out_size,
                    skip_diff=skip_diff_train)
                loss = sum([prior_loss, diff_loss, mel_loss, spk_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 
                                                            max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 
                                                            max_norm=1)
                optim.step()

                logger.add_scalar('training/mel_loss', mel_loss,
                                global_step=iteration)
                logger.add_scalar('training/prior_loss', prior_loss,
                                global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss,
                                global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                global_step=iteration)

                msg = f'Epoch: {epoch}, iteration: {iteration} | ' 
                msg = msg + f'prior_loss: {prior_loss.item():.3f}, '
                msg = msg + f'diff_loss: {diff_loss.item():.3f}, '
                msg = msg + f'mel_loss: {mel_loss.item():.3f}, '
                msg = msg + f'spk_loss: {spk_loss.item():.3f}, '
                progress_bar.set_description(msg)

                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                mel_losses.append(mel_loss.item())
                spk_losses.append(spk_loss.item())
                iteration += 1

        msg = 'Epoch %d: ' % (epoch)
        msg += '| spk loss = %.3f ' % np.mean(spk_losses)
        msg += '| mel loss = %.3f ' % np.mean(mel_losses)
        msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        with open(f'{hps.train.log_dir}/train.log', 'a') as f:
            f.write(msg)
        if (np.mean(prior_losses) < 1.05):
            skip_diff_train = False

        if epoch % hps.train.save_step > 0:
            continue

        save_path = f"{hps.train.log_dir}/grad_svc_{epoch}.pt"
        torch.save({
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'epoch': epoch,
            'steps': iteration,

        }, save_path)
        print("Saved checkpoint to: %s" % save_path)
