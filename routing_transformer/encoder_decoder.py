import re
from inspect import isfunction
import torch
from torch import nn
from routing_transformer.routing_transformer import RoutingTransformerLM, update_kmeans_on_backwards
from routing_transformer.autoregressive_wrapper import AutoregressiveWrapper

ENC_PREFIX = 'enc_'
DEC_PREFIX = 'dec_'

def default(x, d):
    if x is None:
        return d if not isfunction(d) else d()
    return x

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return bool(re.match(f'^{prefix}', str))

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(lambda x: string_begins_with(prefix, x), d)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: string_begins_with(prefix, x), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def extract_enc_dec_kwargs(kwargs):
    enc_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(ENC_PREFIX, kwargs)
    dec_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(DEC_PREFIX, kwargs)
    return enc_kwargs, dec_kwargs, kwargs

def extract_and_set_enc_dec_kwargs(kwargs):
    enc_kwargs, dec_kwargs, kwargs = extract_enc_dec_kwargs(kwargs)
    if 'input_mask' in enc_kwargs:
        dec_kwargs.setdefault('context_mask', enc_kwargs['input_mask'])
    return enc_kwargs, dec_kwargs, kwargs

class RoutingTransformerEncDec(nn.Module):
    def __init__(self, dim, ignore_index = None, pad_value = 0, **kwargs):
        super().__init__()
        ignore_index = default(ignore_index, pad_value)
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)
        
        assert 'return_embedding' not in enc_kwargs, 'you cannot manually set the return embeddings flag for the encoder'
        assert 'dim' not in dec_kwargs and 'dim' not in enc_kwargs, 'you must set the dim for both encoder and decoder'

        enc_kwargs['dim'] = dec_kwargs['dim'] = dim
        enc_kwargs['return_embeddings'] = True
        dec_kwargs['causal'] = True
        dec_kwargs['receives_context'] = True
        enc_kwargs['_register_kmeans_update'] = dec_kwargs['_register_kmeans_update'] = False

        enc_kwargs.setdefault('window_size', 256)
        dec_kwargs.setdefault('window_size', 256)

        enc = RoutingTransformerLM(**enc_kwargs)
        dec = RoutingTransformerLM(**dec_kwargs)

        self.enc = enc
        self.dec = AutoregressiveWrapper(dec, ignore_index = ignore_index, pad_value = pad_value)

        # user will have to manually call backwards on encoder auxiliary loss if the decoder reversibility is turned on
        # should place a bug bounty on this
        self.dec_reversible = dec_kwargs.pop('reversible', False)

        # display a warning message
        if self.dec_reversible:
            print('Warning! Due to an issue with reversible nets and encoder auxiliary losses, you must explicitly call backwards on the encoder auxiliary loss, which is supplied as the second element of the returned tuple on forward')

        update_kmeans_on_backwards(self)

    @torch.no_grad()
    def generate(self, seq_in, seq_out_start, max_seq_len = None, **kwargs):
        max_seq_len = default(max_seq_len, self.dec.max_seq_len)
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        context, _ = self.enc(seq_in, **enc_kwargs)
        return self.dec.generate(seq_out_start, max_seq_len, context = context, **{**dec_kwargs, **kwargs})

    def forward(self, seq_in, seq_out, return_loss = False, randomly_truncate_sequence = False, **kwargs):
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        context, enc_aux_loss = self.enc(seq_in, **enc_kwargs)
        loss = self.dec(seq_out, return_loss = return_loss, randomly_truncate_sequence = randomly_truncate_sequence, context = context, aux_loss = enc_aux_loss, **dec_kwargs)

        # if decoder reversibility turned on, user must manually call backward on encoder auxiliary losses
        if self.dec_reversible:
            return loss, enc_aux_loss

        aux_loss = torch.tensor(0., requires_grad = True)
        loss = loss + enc_aux_loss
        return loss, aux_loss
