import tqdm
import torch
import torch.nn.functional as F


def validate(hp, generator, valloader, writer, step, device):
    generator.eval()
    torch.backends.cudnn.benchmark = False

    loader = tqdm.tqdm(valloader, desc='Validation loop')
    vali_loss = 0.0
    for idx, (phone, phone_l, score, pitch, slurs) in enumerate(loader):
        phone = phone.to(device)
        phone_l = phone_l.to(device)
        score = score.to(device)
        pitch = pitch.to(device)
        slurs = slurs.to(device)

        pitch_pri, pitch_pre = generator(phone, phone_l, score, slurs, n_timesteps=50)

        # De-Log
        pitch_pri = torch.pow(2, pitch_pri)
        pitch_pre = torch.pow(2, pitch_pre)

        loss_f0 = F.l1_loss(pitch_pre[:, 0, :], pitch)
        vali_loss += loss_f0.item()

        if idx < hp.log.num_audio:
            writer.log_fig_pitch(pitch_pri, pitch_pre, pitch, idx, step)

    vali_loss = vali_loss / len(valloader.dataset)
    writer.log_validation(vali_loss, step)

    torch.backends.cudnn.benchmark = True
