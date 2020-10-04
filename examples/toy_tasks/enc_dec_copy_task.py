import tqdm
import torch
import torch.optim as optim

from routing_transformer import RoutingTransformerEncDec

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GENERATE_EVERY  = 100
NUM_TOKENS = 256 + 2
ENC_SEQ_LEN = 128
DEC_SEQ_LEN = 256

# helpers

def cycle():
    while True:
        prefix = torch.ones((BATCH_SIZE, 1)).long().cuda()
        src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long().cuda()
        tgt = torch.cat((prefix, src, src), 1)
        src_mask = torch.ones(BATCH_SIZE, ENC_SEQ_LEN).bool().cuda()
        tgt_mask = torch.ones(BATCH_SIZE, tgt.shape[1]).bool().cuda()
        yield (src, tgt, src_mask, tgt_mask)

# instantiate model

model = RoutingTransformerEncDec(
    dim=512,
    enc_num_tokens=NUM_TOKENS,
    enc_depth=3,
    enc_heads=8,
    enc_max_seq_len=ENC_SEQ_LEN,
    enc_window_size=32,
    dec_num_tokens = NUM_TOKENS,
    dec_depth = 3,
    dec_heads = 8,
    dec_max_seq_len=DEC_SEQ_LEN,
    dec_window_size=32,
).cuda()

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    src, tgt, src_mask, tgt_mask = next(cycle())
    loss, _ = model(src, tgt, enc_input_mask=src_mask, dec_input_mask=tgt_mask, return_loss = True, randomly_truncate_sequence = True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i != 0 and i % GENERATE_EVERY == 0:
        model.eval()
        src, _, src_mask, _ = next(cycle())
        src, src_mask = src[0:1], src_mask[0:1]
        start_tokens = (torch.ones((1, 1)) * 1).long().cuda()

        sample = model.generate(src, start_tokens, ENC_SEQ_LEN, enc_input_mask=src_mask)
        incorrects = (src != sample).abs().sum()

        print(f"input:  ", src)
        print(f"predicted output:  ", sample)
        print(f"incorrects: {incorrects}")
