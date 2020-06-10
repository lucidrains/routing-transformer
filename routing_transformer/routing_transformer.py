import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from inspect import isfunction
from operator import mul
from functools import partial, reduce, wraps

from axial_positional_embedding import AxialPositionalEmbedding
from product_key_memory import PKM
from routing_transformer.reversible import ReversibleSequence, SequentialSequence

# constants

TOKEN_SELF_ATTN_VALUE = -5e4
KMEAN_INIT_ITERS = 10

# helper functions

def identity(x, *args, **kwargs):
    return x

def default(x, d):
    if x is None:
        return d if not isfunction(d) else d()
    return x

def cast_tuple(x):
    return x if isinstance(x, tuple) else (x,)

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def to(t):
    return {'device': t.device, 'dtype': t.dtype}

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

def is_empty(t):
    return t.nelement() == 0

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(2, expand_dim(indices, -1, last_dim))

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def scatter_mean(src, t, index, dim, eps = 1e-5):
    numer = src.scatter_add(dim, index, t)
    denom = src.scatter_add(dim, index, torch.ones_like(t))
    return numer / (denom + eps)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def ema(old, new, decay):
    if old is None:
        return new
    return old * decay + new * (1 - decay)

def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha= (1 - decay))

# helper classes

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks <= 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)

class PreNorm(nn.ModuleList):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class ProjectInOut(nn.Module):
    def __init__(self, fn, dim_in, dim_out, project_out = True):
        super().__init__()
        self.fn = fn
        self.project_in = nn.Linear(dim_in, dim_out)
        self.project_out = nn.Linear(dim_out, dim_in) if project_out else identity

    def forward(self, x, **kwargs):
        x = self.project_in(x)
        x, loss = self.fn(x, **kwargs)
        x = self.project_out(x)
        return x, loss

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)

def shift(x):
    *_, i, j = x.shape
    zero_pad = torch.zeros((*_, i, i), **to(x))
    x = torch.cat([x, zero_pad], -1)
    l = i + j - 1
    x = x.view(*_, -1)
    zero_pad = torch.zeros(*_, -x.size(-1) % l, **to(x))
    shifted = torch.cat([x, zero_pad], -1).view(*_, -1, l)
    return shifted[..., :i, i - 1:]

class RelativePositionalEmbedding(nn.Module):
    def __init__(self, dim, heads, length):
        super().__init__()
        self.scale = dim ** -0.5
        self.weights = nn.Parameter(torch.zeros(length, heads, dim))

    def forward(self, q):
        emb = torch.einsum('bhnid,jhd->bhnij', q, self.weights.type(q.dtype)) * self.scale
        return shift(emb)

# local attention

class LocalAttention(nn.Module):
    def __init__(self, bucket_size, heads, head_dim, causal = False, look_backward = 1, look_forward = None, dropout = 0., shared_qk = False, rel_pos_emb = True):
        super().__init__()
        self.look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and self.look_forward > 0), 'you cannot look forward if causal'

        self.bucket_size = bucket_size
        self.causal = causal
        self.look_backward = look_backward
        self.shared_qk = shared_qk

        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        self.rel_pos = RelativePositionalEmbedding(head_dim, heads, bucket_size * 2) if rel_pos_emb else None

    def forward(self, q, k, v, input_mask = None):
        shape = q.shape

        merge_into_batch = lambda t: t.reshape(-1, *t.shape[-2:])
        q, k, v = map(merge_into_batch, (q, k, v))

        b, t, e, h, device, dtype = *q.shape, self.heads, q.device, q.dtype
        bucket_size, causal, look_backward, look_forward, shared_qk = self.bucket_size, self.causal, self.look_backward, self.look_forward, self.shared_qk

        buckets = t // bucket_size

        if shared_qk:
            k = F.normalize(k, 2, dim=-1).type(q.type())

        ticker = torch.arange(t, device=device, dtype=dtype)[None, :]
        b_t = ticker.reshape(1, buckets, bucket_size)

        bucket_fn = lambda t: t.reshape(b, buckets, bucket_size, -1)
        bq, bk, bv = map(bucket_fn, (q, k, v))

        look_around_kwargs = {'backward': look_backward, 'forward': look_forward}
        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (e ** -0.5)

        if self.rel_pos is not None:
            rel_attn = self.rel_pos(bq.view(-1, h, *bq.shape[1:])).reshape_as(dots)
            dots = dots + rel_attn

        mask_value = max_neg_value(dots)

        if shared_qk:
            mask = bq_t[:, :, :, None] == bq_k[:, :, None, :]
            dots.masked_fill_(mask, TOKEN_SELF_ATTN_VALUE)
            del mask

        if causal:
            mask = bq_t[:, :, :, None] < bq_k[:, :, None, :]
            dots.masked_fill_(mask, mask_value)
            del mask

        mask = bq_k[:, :, None, :] == -1
        dots.masked_fill_(mask, mask_value)
        del mask

        if input_mask is not None:
            h = b // input_mask.shape[0]
            input_mask = input_mask.reshape(-1, buckets, bucket_size)
            mq = mk = input_mask
            mk = look_around(mk, pad_value=False, **look_around_kwargs)
            mask = (mq[:, :, :, None] * mk[:, :, None, :])
            mask = merge_dims(0, 1, expand_dim(mask, 1, h))
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhje->bhie', attn, bv)
        out = out.reshape(*shape)
        return out

# kmeans related function and class

def update_kmeans_on_backwards(module):
    module.kmean_modules = find_modules(module, Kmeans)
    def hook(_, grad_in, grad_out):
        for m in module.kmean_modules:
            m.update()

    return module.register_backward_hook(hook)

def similarity(x, means):
    return torch.einsum('bhld,hcd->bhlc', x, means)

def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets

def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out

def kmeans_iter(x, means, buckets = None):
    b, h, l, d, dtype, num_clusters = *x.shape, x.dtype, means.shape[1]

    if buckets is None:
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_clusters).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, h, num_clusters, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)

    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means

def distribution(dists, window_size):
    _, topk_indices = dists.topk(k=window_size, dim=-2)
    sort_val, _ = topk_indices.sort(dim=-2)
    indices = sort_val.transpose(-2, -1)
    return indices.reshape(*indices.size()[:2], -1)

class Kmeans(nn.Module):
    def __init__(self, num_heads, head_dim, num_clusters, ema_decay = 0.999, commitment = 1e-4):
        super().__init__()
        self.commitment = commitment
        self.ema_decay = ema_decay

        self.register_buffer('means', torch.randn(num_heads, num_clusters, head_dim))
        self.register_buffer('initted', torch.tensor(False))
        self.num_new_means = 0
        self.new_means = None

    @torch.no_grad()
    def init(self, x):
        if self.initted:
            return
        _, h, _, d, device, dtype = *x.shape, x.device, x.dtype

        means = self.means.type(dtype)
        num_clusters = means.shape[1]
        means = x.transpose(0, 1).contiguous().view(h, -1, d)
        indices = torch.randperm(means.size(1), device=device)[:num_clusters]
        means = means[:, indices]

        for _ in range(KMEAN_INIT_ITERS):
            means = kmeans_iter(x, means)

        self.num_new_means = 0
        self.means.data.copy_(means)
        self.initted.data.copy_(torch.tensor(True))

    @torch.no_grad()
    def update(self, new_means = None):
        new_means = default(new_means, self.new_means)
        assert new_means is not None, 'new kmeans has not been supplied'
        ema_inplace(self.means, new_means, self.ema_decay)

        del self.new_means
        self.new_means = None
        self.num_new_means = 0

    def forward(self, x, update_means = False):
        self.init(x)

        b, dtype = x.shape[0], x.dtype
        means = self.means.type(dtype)
        x = F.normalize(x, 2, dim=-1).type(dtype)

        with torch.no_grad():
            dists, buckets = dists_and_buckets(x, means)

        routed_means = batched_index_select(expand_dim(means, 0, b), buckets)
        loss = F.mse_loss(x, routed_means) * self.commitment

        if update_means:
            with torch.no_grad():
                means = kmeans_iter(x, means, buckets)
            self.new_means = ema(self.new_means, means, self.num_new_means / (self.num_new_means + 1))
            self.num_new_means += 1

        return dists, loss

# kmeans attention class

class KmeansAttention(nn.Module):
    def __init__(self, num_clusters, window_size, num_heads, head_dim, causal = False, dropout = 0., ema_decay = 0.999, commitment = 1e-4, context_window_size = None, shared_qk = True):
        super().__init__()
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.head_dim = head_dim

        self.window_size = window_size
        self.context_window_size = default(context_window_size, window_size)
        self.causal = causal

        self.shared_qk = shared_qk
        self.kmeans = Kmeans(num_heads, head_dim, num_clusters, ema_decay, commitment)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, query_mask = None, key_mask = None, **kwargs):
        b, h, t, d, kv_t, wsz, c_wsz, nc, device, dtype = *q.shape, k.shape[2], self.window_size, self.context_window_size, self.num_clusters, q.device, q.dtype
        is_reverse = kwargs.pop('_reverse', False)

        out = torch.zeros_like(q, dtype=dtype)

        update_kmeans = self.training and not is_reverse

        if self.shared_qk:
            wsz = min(wsz, t)
            dists, aux_loss = self.kmeans(k, update_kmeans)
            indices = distribution(dists, wsz)
            k = F.normalize(k, 2, dim=-1).type(dtype)
            key_mask = query_mask
            kv_indices = indices
        else:
            wsz = min(wsz, t)
            c_wsz = min(c_wsz, kv_t)

            dists, aux_loss = self.kmeans(torch.cat((q, k), dim=2), update_kmeans)
            q_dists, k_dists = split_at_index(2, t, dists)
            indices = distribution(q_dists, wsz)
            kv_indices = distribution(k_dists, c_wsz)

        q = batched_index_select(q, indices)
        k = batched_index_select(k, kv_indices)
        v = batched_index_select(v, kv_indices)

        reshape_with_window = lambda x: x.reshape(b, h, nc, -1, d)
        q, k, v = map(reshape_with_window, (q, k, v))

        dots = torch.einsum('bhnid,bhnjd->bhnij', q, k) * (d ** -0.5)

        mask_value = max_neg_value(dots)

        if query_mask is not None or key_mask is not None:
            query_mask = default(query_mask, lambda: torch.ones((b, t), device=device).bool())
            key_mask = default(key_mask, lambda: torch.ones((b, kv_t), device=device).bool())

            q_mask = expand_dim(query_mask, 1, h).gather(2, indices)
            kv_mask = expand_dim(key_mask, 1, h).gather(2, kv_indices)
            q_mask, kv_mask = map(lambda t: t.reshape(b, h, nc, -1), (q_mask, kv_mask))
            mask = q_mask[:, :, :, :, None] * kv_mask[:, :, :, None, :]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.causal:
            mask = torch.ones(wsz, wsz, device=device).byte().triu_(1).bool()
            dots.masked_fill_(mask, mask_value)
            del mask

        if self.shared_qk:
            mask = torch.eye(wsz, device=dots.device).bool()
            dots.masked_fill_(mask, TOKEN_SELF_ATTN_VALUE)
            del mask

        dots = dots.softmax(dim=-1)
        dots = self.dropout(dots)

        bo = torch.einsum('bhcij,bhcjd->bhcid', dots, v)
        so = torch.reshape(bo, (b, h, -1, bo.shape[-1])).type(dtype)
        out = scatter_mean(out, so, indices.unsqueeze(-1).expand_as(so), -2)
        return out, aux_loss

# feedforward

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

# self attention

class SelfAttention(nn.Module):
    def __init__(self,  dim, depth, max_seq_len, heads, local_attn_heads, window_size, local_attn_window_size = None, causal = False, attn_dropout = 0., dropout = 0., kmeans_ema_decay = 0.999, commitment_factor = 1e-4, receives_context = False, context_window_size = None, rel_pos_emb = True):
        super().__init__()
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        assert (max_seq_len % window_size) == 0, 'maximum sequence length must be divisible by the target window size'
        assert local_attn_heads <= heads, 'number of local attention heads must be less than total heads'
        assert not (receives_context and local_attn_heads > 0), 'local attention cannot be used for self attention with context'
        assert not (receives_context and causal), 'contextual attention layer cannot be causal'

        local_attn_window_size = default(local_attn_window_size, window_size // 2)
        context_window_size = default(context_window_size, window_size)

        self.receives_context = receives_context
        self.heads = heads
        self.local_attn_heads = local_attn_heads
        self.global_attn_heads = heads - local_attn_heads

        self.window_size = window_size

        head_dim = dim // heads
        num_clusters = max_seq_len // window_size

        if self.local_attn_heads > 0:
            self.local_attn = LocalAttention(local_attn_window_size, local_attn_heads, head_dim, causal = True, shared_qk = True, dropout = attn_dropout, rel_pos_emb = rel_pos_emb)

        if self.global_attn_heads > 0:
            self.global_attn = KmeansAttention(num_clusters, window_size, self.global_attn_heads, head_dim, causal = causal, dropout = attn_dropout, ema_decay = kmeans_ema_decay, commitment = commitment_factor, shared_qk = not receives_context)

        self.to_qk = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)
        self.to_out = nn.Linear(dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None, input_mask = None, context_mask = None, **kwargs):
        assert not (self.receives_context and context is None), 'context must be passed if self attention is set to receive context'
        b, t, e, h = *x.shape, self.heads
        head_dim = e // self.heads

        split_heads = lambda v: v.reshape(b, -1, h, head_dim).transpose(1, 2).contiguous()

        if not self.receives_context:
            qk, v = self.to_qk(x), self.to_v(x)

            qk, v = map(split_heads, (qk, v))
            split_index_fn = partial(split_at_index, 1, self.local_attn_heads)
            (lqk, qk), (lv, v) = map(split_index_fn, (qk, v))
            has_local, has_global = map(lambda x: x.shape[1] > 0, (lqk, qk))

            q, k, v = qk, qk, v
        else:
            q, k, v = self.to_qk(x), self.to_qk(context), self.to_v(context)
            q, k, v = map(split_heads, (q, k, v))
            has_local, has_global = False, True

        out = []
        total_loss = torch.tensor(0., requires_grad=True, **to(x))

        if has_local:
            local_out = self.local_attn(lqk, lqk, lv, input_mask = input_mask)
            out.append(local_out)

        if has_global:
            global_out, loss = self.global_attn(q, k, v, query_mask = input_mask, key_mask = context_mask)
            total_loss = total_loss + loss

            out.append(global_out)

        out = torch.cat(out, dim=1)
        out = out.reshape(b, h, t, -1).transpose(1, 2).reshape(b, t, -1)
        out = self.to_out(out)
        return self.dropout(out), total_loss

class RoutingTransformer(nn.Module):
    def __init__(self, dim, depth, max_seq_len, heads = 8, window_size = 64, local_attn_window_size = None, causal = False, weight_tie = False, attn_dropout = 0., ff_dropout = 0., attn_layer_dropout = 0., layer_dropout = 0., n_local_attn_heads = 0, ff_glu = False, reversible = False, ff_chunks = 1, kmeans_ema_decay = 0.999, commitment_factor = 1e-4, receives_context = False, context_window_size = None, _register_kmeans_update = False, rel_pos_emb = True, pkm_layers = tuple(), pkm_num_keys = 128):
        super().__init__()
        local_attn_window_size = default(local_attn_window_size, window_size // 2)
        if type(n_local_attn_heads) is not tuple:
            n_local_attn_heads = tuple([n_local_attn_heads] * depth)

        assert len(n_local_attn_heads) == depth, 'local attention heads tuple must have the same length as the depth'
        assert all([(local_heads <= heads) for local_heads in n_local_attn_heads]), 'number of local attn heads must be less than the maximum number of heads'

        layers = nn.ModuleList([])
        fn_wrapper = partial(PreNorm, dim)

        get_attn = lambda local_heads: SelfAttention(dim, depth, max_seq_len, heads, local_heads, window_size, causal = causal, local_attn_window_size = local_attn_window_size, attn_dropout = attn_dropout, dropout = attn_layer_dropout, kmeans_ema_decay = kmeans_ema_decay, commitment_factor = commitment_factor, rel_pos_emb = rel_pos_emb)
        get_ff = lambda: Chunk(ff_chunks, FeedForward(dim, dropout = ff_dropout, glu = ff_glu), along_dim=1)
        get_context_attn = lambda: SelfAttention(dim, depth, max_seq_len, heads, 0, window_size, local_attn_window_size = local_attn_window_size, attn_dropout = attn_dropout, dropout = attn_layer_dropout, kmeans_ema_decay = kmeans_ema_decay, commitment_factor = commitment_factor, receives_context = True, context_window_size = context_window_size)
        get_context_ff = lambda: Chunk(ff_chunks, FeedForward(dim, dropout = ff_dropout, glu = ff_glu), along_dim=1)
        get_pkm = lambda: PKM(dim, num_keys = pkm_num_keys, use_evonorm = False)

        if weight_tie:
            assert len(set(n_local_attn_heads)) == 1, 'you can only weight tie if number of local attention heads for all layers is the same'
            get_attn, get_ff, get_context_attn, get_context_ff, get_pkm = map(cache_fn, (get_attn, get_ff, get_context_attn, get_context_ff, get_pkm))

        for ind, local_heads in zip(range(depth), n_local_attn_heads):
            layer = ind + 1
            use_ff = layer not in cast_tuple(pkm_layers)
            get_parallel_fn = get_ff if use_ff else get_pkm

            layers.append(nn.ModuleList([
                fn_wrapper(get_attn(local_heads)),
                fn_wrapper(get_parallel_fn())
            ]))

            if not receives_context:
                continue

            layers.append(nn.ModuleList([
                fn_wrapper(get_context_attn()),
                fn_wrapper(get_context_ff())
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        attn_context_layer = ((True, False),) if receives_context else tuple()
        route_attn = ((True, False), *attn_context_layer) * depth
        route_context = ((False, False), *attn_context_layer) * depth

        context_route_map = {'context': route_context, 'context_mask': route_context} if receives_context else {}
        attn_route_map = {'input_mask': route_attn}
        self.layers = execute_type(layers, args_route = {**attn_route_map, **context_route_map}, layer_dropout = layer_dropout)

        if _register_kmeans_update:
            update_kmeans_on_backwards(self)

        has_local_attn = any([num > 0 for num in n_local_attn_heads])
        self.pad_to_multiple = local_attn_window_size if has_local_attn else 0

    def forward(self, x, **kwargs):
        x, loss = self.layers(x, **kwargs)
        return x, loss

class RoutingTransformerLM(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads = 8, window_size = 64, local_attn_window_size = None, causal = False, emb_dim = None, weight_tie = False, attn_dropout = 0., ff_dropout = 0., attn_layer_dropout = 0., layer_dropout = 0., ff_mult = 4, ff_activation = None, ff_glu = False, return_embeddings = False, n_local_attn_heads = 0, reversible = False, ff_chunks = 1, kmeans_ema_decay = 0.999, commitment_factor = 1e-4, receives_context = False, context_window_size = None, rel_pos_emb = True, _register_kmeans_update = True, pkm_layers = tuple(), pkm_num_keys = 128):
        super().__init__()
        assert (max_seq_len % window_size) == 0, 'max sequence length must be divisible by the window size, to calculate number of kmeans cluster'
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.axial_pos_emb = AxialPositionalEmbedding(emb_dim, axial_shape=(max_seq_len // window_size, window_size))
        self.routing_transformer = RoutingTransformer(dim, depth, max_seq_len, heads = heads, window_size = window_size, local_attn_window_size = local_attn_window_size, causal = causal, weight_tie = weight_tie, ff_dropout = ff_dropout, attn_dropout = attn_dropout, attn_layer_dropout = attn_layer_dropout, layer_dropout = layer_dropout, n_local_attn_heads = n_local_attn_heads, ff_glu = ff_glu, reversible = reversible, ff_chunks = ff_chunks, kmeans_ema_decay = kmeans_ema_decay, receives_context = receives_context, context_window_size = context_window_size, rel_pos_emb = rel_pos_emb, pkm_layers = pkm_layers, pkm_num_keys = pkm_num_keys, _register_kmeans_update = _register_kmeans_update)

        if emb_dim != dim:
            self.routing_transformer = ProjectInOut(self.routing_transformer, emb_dim, dim, project_out = not return_embeddings)

        self.out = nn.Linear(emb_dim, num_tokens) if not return_embeddings else identity

    def forward(self, x, **kwargs):
        x = self.token_emb(x)
        x = x + self.axial_pos_emb(x)
        x, loss = self.routing_transformer(x, **kwargs)
        return self.out(x), loss
