## Routing Transformer (wip)

<img src="./routing_attention.png" width="500px"></img>

[![PyPI version](https://badge.fury.io/py/routing-transformer.svg)](https://badge.fury.io/py/routing-transformer)

A fully featured implementation of <a href="https://arxiv.org/pdf/2003.05997.pdf">Routing Transformer</a>. The paper proposes using k-nearest neighbors to route queries / keys into clusters for attention with one another.

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
    causal = True,	        # auto-regressive or not
    window_size = 128,      # target window size of each cluster
    n_local_attn_heads = 4  # number of local attention heads
).cuda()

x = torch.randint(0, 20000, (1, 8192)).long().cuda()
model(x) # (1, 8192, 20000)
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
model(x) # (1, 8192, 512)
```

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
