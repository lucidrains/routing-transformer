## Routing Transformer (wip)

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
input_mask = torch.ones_like(x).bool().cuda()

y, aux_loss = model(x, input_mask = input_mask) # (1, 8192, 512)
aux_loss.backward() # add auxiliary loss to main loss before backprop
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