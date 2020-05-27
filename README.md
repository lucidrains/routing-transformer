## Routing Transformer

<img src="./routing_attention.png" width="500px"></img>

[![PyPI version](https://badge.fury.io/py/routing-transformer.svg)](https://badge.fury.io/py/routing-transformer)

A fully featured implementation of <a href="https://arxiv.org/pdf/2003.05997.pdf">Routing Transformer</a>. The paper proposes using k-nearest neighbors to route similar queries / keys into the same cluster for attention.

### Install

```bash
$ pip install routing_transformer
```

### Usage

A simple language model

```python
import torch
from routing_transformer import RoutingTransformerLM

model = RoutingTransformerLM(
    num_tokens = 20000,
    dim = 512,
    heads = 8,
    depth = 12,
    max_seq_len = 8192,
    causal = True,           # auto-regressive or not
    emb_dim = 128,           # embedding factorization
    attn_dropout = 0.1,      # dropout after attention
    attn_layer_dropout = 0., # dropout after self attention layer
    ff_dropout = 0.1,        # feedforward dropout
    layer_dropout = 0.,      # layer dropout
    window_size = 128,       # target window size of each cluster
    n_local_attn_heads = 4,  # number of local attention heads
    reversible = True,       # reversible networks for memory savings, from Reformer paper
    ff_chunks = 10,          # feed forward chunking, from Reformer paper
    ff_glu = True,           # use GLU variant in feedforward
).cuda()

x = torch.randint(0, 20000, (1, 8192)).long().cuda()
input_mask = torch.ones_like(x).bool().cuda()

y, aux_loss = model(x, input_mask = input_mask) # (1, 8192, 20000)
aux_loss.backward() # add auxiliary loss to main loss before backprop
```

A simple transformer

```python
import torch
from routing_transformer import RoutingTransformer

model = RoutingTransformer(
    dim = 512,
    heads = 8,
    depth = 12,
    max_seq_len = 8192,
    window_size = 128,
    n_local_attn_heads = 4
).cuda()

x = torch.randn(1, 8192, 512).cuda()
input_mask = torch.ones(1, 8192).bool().cuda()

y, aux_loss = model(x, input_mask = input_mask) # (1, 8192, 512)
aux_loss.backward() # add auxiliary loss to main loss before backprop
```

Encoder / decoder

```python
enc = RoutingTransformerLM(
    num_tokens = 20000,
    dim = 512,
    heads = 8,
    depth = 6,
    max_seq_len = 4096,
    window_size = 128,
    n_local_attn_heads = 4,
    return_embeddings = True
).cuda()

dec = RoutingTransformerLM(
    num_tokens = 20000,
    dim = 512,
    heads = 8,
    depth = 6,
    causal = True,
    max_seq_len = 4096,
    window_size = 128,
    n_local_attn_heads = 4,
    receives_context = True
).cuda()

src = torch.randint(0, 20000, (1, 4096)).cuda()
tgt = torch.randint(0, 20000, (1, 4096)).cuda()

src_mask = torch.ones_like(src).bool().cuda()
tgt_mask = torch.ones_like(tgt).bool().cuda()

context, enc_aux_loss = enc(src, input_mask = src_mask)
out, dec_aux_loss = dec(tgt, context = context, context_mask = src_mask, input_mask = tgt_mask)
```

## Kmeans Hyperparameters

1. `kmeans_ema_decay = {defaults to 0.999}`

This is the exponential moving average decay for updating the k-means. The lower this is, the faster the means will adjust, but at the cost of stability.

2. `commitment_factor = {defaults to 1e-4}`

The weight of the auxiliary loss that encourages tokens to get closer (commit) to the k-mean centroids that were chosen for them.

## Appreciation

Special thanks to <a href="https://github.com/AranKomat">Aran Komatsuzaki</a> for bootstrapping the initial implementation in Pytorch that evolved into this library.

## Citation

```bibtex
@misc{roy*2020efficient,
    title   = {Efficient Content-Based Sparse Attention with Routing Transformers},
    author  = {Aurko Roy* and Mohammad Taghi Saffar* and David Grangier and Ashish Vaswani},
    year    = {2020},
    url     = {https://arxiv.org/pdf/2003.05997.pdf}
}
```

```bibtex
@misc{shazeer2020glu,
    title   = {GLU Variants Improve Transformer},
    author  = {Noam Shazeer},
    year    = {2020},
    url     = {https://arxiv.org/abs/2002.05202}    
}
```

```bibtex
@inproceedings{kitaev2020reformer,
    title       = {Reformer: The Efficient Transformer},
    author      = {Nikita Kitaev and Lukasz Kaiser and Anselm Levskaya},
    booktitle   = {International Conference on Learning Representations},
    year        = {2020},
    url         = {https://openreview.net/forum?id=rkgNKkHtvB}
}
```

```bibtex
@inproceedings{fan2020reducing,
    title     ={Reducing Transformer Depth on Demand with Structured Dropout},
    author    ={Angela Fan and Edouard Grave and Armand Joulin},
    booktitle ={International Conference on Learning Representations},
    year      ={2020},
    url       ={https://openreview.net/forum?id=SylO2yStDr}
}
```

```bibtex
@misc{lan2019albert,
    title       = {ALBERT: A Lite BERT for Self-supervised Learning of Language Representations},
    author      = {Zhenzhong Lan and Mingda Chen and Sebastian Goodman and Kevin Gimpel and Piyush Sharma and Radu Soricut},
    year        = {2019},
    url         = {https://arxiv.org/abs/1909.11942}
}
```