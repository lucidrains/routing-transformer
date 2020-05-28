import math
import torch
from torch import nn
from routing_transformer.routing_transformer import RoutingTransformer
import torch.nn.functional as F

def find_module(nn_module, type):
    for module in nn_module.modules():
        if isinstance(module, type):
            return module
    return None

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
    def __init__(self, net):
        super().__init__()
        transformer = find_module(net, RoutingTransformer)
        self.net = net
        self.pad_multiple = transformer.pad_to_multiple

    def forward(self, x, **kwargs):
        if self.pad_multiple <= 0:
            return self.net(x, **kwargs)

        b, t, device = *x.shape, x.device

        input_mask = kwargs.get('input_mask')

        if input_mask is None:
            input_mask = torch.full((b, t), True, device=device, dtype=torch.bool)

        x = pad_to_multiple(x, self.pad_multiple, dim=1)
        new_mask = pad_to_multiple(input_mask, self.pad_multiple, dim=1, value=False)
        kwargs.update(input_mask=new_mask)

        out, loss = self.net(x, **kwargs)
        return out[:, 0:t], loss
