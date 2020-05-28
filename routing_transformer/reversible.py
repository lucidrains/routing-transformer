import torch
import torch.nn as nn
from operator import itemgetter
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

# for routing arguments into the functions of the reversible layer

def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args

def layer_drop(layers, prob):
    to_drop = torch.empty(len(layers)).uniform_(0, 1) < prob
    blocks = [block for block, drop in zip(layers, to_drop) if not drop]
    blocks = layers[:1] if len(blocks) == 0 else blocks
    return blocks

def cast_return(ret, requires_grad = True):
    if type(ret) is not tuple:
        loss = torch.tensor(0., device=ret.device, dtype=ret.dtype, requires_grad=requires_grad)
        return (ret, loss)
    return ret

# following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng = False, set_rng = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source
class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args = {}, g_args = {}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None

        f_args['_reverse'] = g_args['_reverse'] = False

        with torch.no_grad():
            f_out, f_loss = cast_return(self.f(x2, record_rng=self.training, **f_args), requires_grad = False)
            y1 = x1 + f_out

            g_out, g_loss = cast_return(self.g(y1, record_rng=self.training, **g_args), requires_grad = False)
            y2 = x2 + g_out

        return torch.cat([y1, y2], dim=2), f_loss, g_loss

    def backward_pass(self, y, dy, dl_f, dl_g, f_args = {}, g_args = {}):
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy

        f_args['_reverse'] = g_args['_reverse'] = True

        with torch.enable_grad():
            y1.requires_grad = True
            gy1, g_loss = cast_return(self.g(y1, set_rng=True, **g_args))
            torch.autograd.backward((gy1, g_loss), (dy2, dl_g))

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2, f_loss = cast_return(self.f(x2, set_rng=True, **f_args))
            torch.autograd.backward((fx2, f_loss), (dx1, dl_f), retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)

        return x, dx

class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, args):
        ctx.args = args

        f_aux_loss = []
        g_aux_loss = []

        for block, kwarg in zip(blocks, args):
            x, f_loss, g_loss = block(x, **kwarg)
            f_aux_loss.append(f_loss)
            g_aux_loss.append(g_loss)

        ctx.y = x.detach()
        ctx.blocks = blocks
        return x, torch.stack(f_aux_loss), torch.stack(g_aux_loss)

    @staticmethod
    def backward(ctx, dy, dl_f, dl_g):
        y = ctx.y
        args = ctx.args
        for block, kwargs, ind in zip(ctx.blocks[::-1], args[::-1], range(len(ctx.blocks))[::-1]):
            y, dy = block.backward_pass(y, dy, dl_f[ind], dl_g[ind], **kwargs)
        return dy, None, None

class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route = {}, layer_dropout = 0.):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route
        self.layer_dropout = layer_dropout

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        if self.training and self.layer_dropout > 0:
            layers_and_args = layer_drop(layers_and_args, self.layer_dropout)

        aux_loss = torch.zeros(1, device=x.device, dtype=x.dtype)

        for (f, g), (f_args, g_args) in layers_and_args:
            res, loss = cast_return(f(x, **f_args))
            aux_loss += loss
            x = x + res

            res, loss = cast_return(g(x, **g_args))
            aux_loss += loss
            x = x + res
        return x, aux_loss

class ReversibleSequence(nn.Module):
    def __init__(self, blocks, args_route = {}, layer_dropout = 0.):
        super().__init__()
        self.args_route = args_route
        self.layer_dropout = layer_dropout
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for f, g in blocks])

    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim=-1)

        blocks = self.blocks
        args = route_args(self.args_route, kwargs, len(blocks))
        args = list(map(lambda x: {'f_args': x[0], 'g_args': x[1]}, args))

        layers_and_args = list(zip(blocks, args))

        if self.training and self.layer_dropout > 0:
            layers_and_args = layer_drop(layers_and_args, self.layer_dropout)
            blocks, args = map(lambda ind: list(map(itemgetter(ind), layers_and_args)), (0, 1))

        out, f_loss, g_loss =  _ReversibleFunction.apply(x, blocks, args)
        out = torch.stack(out.chunk(2, dim=-1)).mean(dim=0)
        aux_loss = f_loss.sum() + g_loss.sum()
        return out, aux_loss
