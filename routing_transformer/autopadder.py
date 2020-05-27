import math
import torch
from torch import nn
import torch.nn.functional as F

def pad_to_multiple(tensor, multiple, dim=-1):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return tensor, 0

    pre_pad_offset = (0,) * (-1 - dim) * 2
    padding = math.ceil(m) * multiple - seqlen
    padded_tensor = F.pad(tensor, (*pre_pad_offset, *(0, padding)), value=0)
    return padded_tensor, padding

class Autopadder(nn.Module):
    def __init__(self, net, pad_multiple, dim=-2):
        super().__init__()
        self.net = net
        self.pad_multiple = pad_multiple
        self.pad_dim = dim

    def forward(self, *args, **kwargs):
        args = list(args)

        x = args[0]
        b, _, t, _, device = *x.shape, x.device

        input_mask = kwargs.get('input_mask')

        if input_mask is None:
            input_mask = torch.full_like(x, True, device=device, dtype=torch.bool)

        for ind, x in enumerate(args):
            x, padding = pad_to_multiple(x, self.pad_multiple, dim=self.pad_dim)
            args[ind] = x

        if padding != 0:
            new_mask = F.pad(input_mask, (0, padding), value=False)
            kwargs.update(input_mask=new_mask)

        out = self.net(*args, **kwargs)
        return out[:, :, 0:t]
