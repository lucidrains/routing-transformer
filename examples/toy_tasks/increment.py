import torch
import numpy as np
import math
import time
import random
from torch.optim import Adam
from routing_transformer.routing_transformer import RoutingTransformerLM
from routing_transformer.autoregressive_wrapper import AutoregressiveWrapper

s = RoutingTransformerLM(
    num_tokens = 256 + 4,
    dim = 1024,
    depth = 2,
    heads = 8,
    max_seq_len = 256,
    causal = True,
    window_size = 128
).cuda()

s = AutoregressiveWrapper(s, ignore_index = 0, pad_value = 0)
opt = Adam(s.parameters(), lr=1e-4)

N_BATCH = 32
SRC_SEQ_LEN = 128
TGT_SEQ_LEN = 128

bos = 1*torch.ones(N_BATCH, 1).long()
eos = 2*torch.ones(N_BATCH, 1).long()
pos = 3*torch.ones(N_BATCH, 1).long()

for i in range(10000):
    train_seq_in = torch.randint(4, 6, (N_BATCH, SRC_SEQ_LEN - 2)).long()
    train_seq_out = train_seq_in + 1

    train_seq = torch.cat([bos, train_seq_in, pos, pos, pos, train_seq_out, eos], dim=1).cuda()

    loss = s(train_seq, return_loss = True)
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(i, loss.item())