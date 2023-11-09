from torch.utils.tensorboard import SummaryWriter
from .plotting import plot_f0_to_numpy


class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)

    def log_training(self, loss_g, prior_loss, diff_loss, step):
        self.add_scalar('train/loss_g', loss_g, step)
        self.add_scalar('train/loss_prior', prior_loss, step)
        self.add_scalar('train/loss_diff', diff_loss, step)

    def log_validation(self, vali_loss, step):
        self.add_scalar('validation/vali_loss', vali_loss, step)

    def log_fig_pitch(self, pitch_prio, pitch_fake, pitch_real, idx, step):
        if idx == 0:
            pitch_prio = pitch_prio[0, 0, :].data.cpu().numpy()
            pitch_fake = pitch_fake[0, 0, :].data.cpu().numpy()
            pitch_prio[pitch_prio > 1000] = 1000
            pitch_fake[pitch_fake > 1000] = 1000
            pitch_real = pitch_real[0].data.cpu().numpy()
            self.add_image(f'pitch_prio/{step}', plot_f0_to_numpy(pitch_prio, pitch_real), step)
            self.add_image(f'pitch_fake/{step}', plot_f0_to_numpy(pitch_fake, pitch_real), step)
            # self.add_image(f'pitch_real/{step}', plot_f0_to_numpy(pitch_real), step)
