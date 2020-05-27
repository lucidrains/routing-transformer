import math
import torch
from torch import nn
import torch.nn.functional as F

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return tensor

    pre_pad_offset = (0,) * (-1 - dim) * 2
    padding = math.ceil(m) * multiple - seqlen
    padded_tensor = F.pad(tensor, (*pre_pad_offset, *(0, padding)), value=value)
    return padded_tensor

class Autopadder(nn.Module):
    def __init__(self, net, pad_multiple):
        super().__init__()
        self.net = net
        self.pad_multiple = pad_multiple

    def forward(self, *args, **kwargs):
        q, args = args[0], list(args)
        b, h, t, _, device = *q.shape, q.device

        input_mask = kwargs.get('input_mask')

        if input_mask is None:
            input_mask = torch.full((b, t), True, device=device, dtype=torch.bool)

        args = map(lambda t: pad_to_multiple(t, self.pad_multiple, dim=-2), args)
        new_mask = pad_to_multiple(input_mask, self.pad_multiple, dim=-1, value=False)
        kwargs.update(input_mask=new_mask)

        out = self.net(*args, **kwargs)
        return out[:, :, 0:t]
