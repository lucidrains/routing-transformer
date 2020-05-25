import math
import torch
from torch import nn
import torch.nn.functional as F
from routing_transformer.routing_transformer import RoutingTransformer, RoutingTransformerLM

def find_module(nn_module, type):
    for module in nn_module.modules():
        if isinstance(module, type):
            return module
    return None

def pad_to_multiple(tensor, multiple, dim=-1, pad_left = False):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return tensor, 0

    pre_pad_offset = (0,) * (-1 - dim) * 2
    padding = math.ceil(m) * multiple - seqlen
    offset = (padding, 0) if pad_left else (0, padding)
    padded_tensor = F.pad(tensor, (*pre_pad_offset, *offset), value=0)
    return padded_tensor, padding

class Autopadder(nn.Module):
    def __init__(self, net, pad_left=False):
        super().__init__()
        assert isinstance(net, (RoutingTransformer, RoutingTransformerLM)), 'only modules RoutingTransformer and RoutingTransformerLM accepted'
        self.net = net

        is_lm = isinstance(net, RoutingTransformerLM)
        routing_transformer = find_module(net, RoutingTransformer)
        self.bucket_size = routing_transformer.pad_to_window_size
        self.context_bucket_size = routing_transformer.pad_to_window_size

        self.pad_dim = -1 if is_lm else -2
        self.pad_left = pad_left

    def forward(self, x, **kwargs):
        b, t, device = *x.shape[:2], x.device

        context = kwargs.get('context')
        input_mask = kwargs.get('input_mask')
        context_mask = kwargs.get('context_mask')

        if input_mask is None:
            input_mask = torch.full_like(x, True, device=x.device, dtype=torch.bool)

        if context is not None and context_mask is None:
            context_mask = torch.full(context.shape[0:2], True, device=x.device, dtype=torch.bool)

        x, padding = pad_to_multiple(x, self.bucket_size, dim=self.pad_dim, pad_left=self.pad_left)

        if padding != 0:
            offset = (0, padding) if not self.pad_left else (padding, 0)
            new_mask = F.pad(input_mask, offset, value=False)
            kwargs.update(input_mask=new_mask)

        if context is not None:
            context, context_padding = pad_to_multiple(context, self.context_bucket_size, dim=-2)

            if context_padding != 0:
                new_mask = F.pad(context_mask, (0, context_padding), value=False)
                kwargs.update(context_mask=new_mask)

            kwargs.update(context=context)

        out, loss = self.net(x, **kwargs)

        output_slice = slice(0, t) if not self.pad_left else slice(padding, None)
        return out[:, output_slice], loss
