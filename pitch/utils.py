import torch
import inspect


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], 
                                          [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_)**2) / torch.sum(lengths)
    return loss


def rand_ids_segments(lengths, segment_size=200):
    b = lengths.shape[0]
    ids_str_max = lengths - segment_size
    ids_str = (torch.rand([b]).to(device=lengths.device) * ids_str_max).to(dtype=torch.long)
    ids_str = torch.where(ids_str < 0, 0, ids_str)  # fix error
    return ids_str


def slice_segments(x, ids_str, segment_size=200):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def retrieve_name(var):
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name,
                 var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


Debug_Enable = True


def debug_shapes(var):
    if Debug_Enable:
        print(retrieve_name(var), var.shape)
