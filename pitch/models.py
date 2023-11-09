import math
import torch
import torch.nn.functional as F

from torch import nn
from pitch.diffusion import Diffusion
from pitch.utils import rand_ids_segments, slice_segments

from vits import attentions
from vits import commons


class TextEncoder(nn.Module):
    def __init__(self,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.emb_phone = nn.Embedding(63, hidden_channels)      # phone lables
        self.emb_score = nn.Embedding(128, hidden_channels)     # pitch notes
        self.emb_slurs = nn.Embedding(2, hidden_channels)       # phone slur
        nn.init.normal_(self.emb_phone.weight, 0.0, hidden_channels**-0.5)
        nn.init.normal_(self.emb_score.weight, 0.0, hidden_channels**-0.5)
        nn.init.normal_(self.emb_slurs.weight, 0.0, hidden_channels**-0.5)
        self.enc = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, 2, 1)  # pitch + uv

    def forward(self, phone, lengths, score, slurs):
        x = self.emb_phone(phone) + self.emb_score(score) + self.emb_slurs(slurs)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        c = x
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.enc(x * x_mask, x_mask)
        x = self.proj(x)
        return x, x_mask, c


class PitchDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.pit_encoder = TextEncoder(hidden_channels=192, filter_channels=768, 
                                       n_heads=2, n_layers=5, kernel_size=5, p_dropout=0.1)
        self.decoder = Diffusion(2, 64, 192, beta_min=0.05, beta_max=20.0, pe_scale=1000)


    @torch.no_grad()
    def forward(self, phone, lengths, score, slurs, n_timesteps, temperature=1.0, stoc=False):
        # Encoder
        mu_x, mask_x, c = self.pit_encoder(phone, lengths, score, slurs)
        encoder_outputs = mu_x

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_x + torch.randn_like(mu_x, device=mu_x.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(c, z, mask_x, mu_x, n_timesteps, stoc)
        return encoder_outputs, decoder_outputs

    def compute_loss(self, phone, lengths, score, slurs, pitch, out_size):
        # Get encoder_outputs `mu_x`
        mu_x, mask_x, c = self.pit_encoder(phone, lengths, score, slurs)

        # Compute loss between encoder outputs and pitch
        floor = torch.ones_like(pitch)
        pitch = torch.maximum(pitch, floor)
        pitch = torch.log2(pitch)
        # Loss
        loss_f0 = F.l1_loss(mu_x[:, 0, :], pitch)
        uv_gt = (pitch > 0).to(pitch.dtype)
        loss_uv = F.binary_cross_entropy_with_logits(mu_x[:, 1, :], uv_gt)
        prior_loss = loss_f0 + loss_uv
        # pitch_gt
        pitch_gt = torch.zeros_like(mu_x, device=mu_x.device)
        pitch_gt[:, 0, :] = pitch
        pitch_gt[:, 1, :] = uv_gt
        # Compute loss of score-based decoder
        # Cut a small segment of pitch in order to increase batch size
        if not isinstance(out_size, type(None)) and out_size < pitch_gt.shape[1]:
            ids = rand_ids_segments(lengths, out_size)
            pitch_gt = slice_segments(pitch_gt, ids, out_size)

            mask_x = slice_segments(mask_x, ids, out_size)
            mu_x = slice_segments(mu_x, ids, out_size)
            c = slice_segments(c, ids, out_size)

        diff_loss, xt = self.decoder.compute_loss(c, pitch_gt, mask_x, mu_x)
        return prior_loss, diff_loss
 
