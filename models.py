"""
Contains adaptations of existing VALL-E implementation from:
https://github.com/enhuiz/vall-e

(Base) Components:
- init()
- [x] SinusodialEmbedding (2017 original positional embedding)
- Attention (MHSA?, Single head attention?)
- AdaLN (Adaptive Layer Norm, better for generative networks)
- PrenormResidual (Norm before attn/ffwd), opposite of 2017 paper
- Block (TransformerBlock)
- Embedding (Generic forward pass for an embedding)
- MultiEmbedding (Sum embeddings on different levels)

(AR) Components:
- init()
- forward()
- generate()

```python
qnt     = _ # quant codes?
model   = AR(num_quants).to(device)
txt_lst = [
    tensor([1, 2, 3], device=device),
    tensor([2, 3], device=device)
]
x8 = partial(repeat, pattern="1 -> t 1", l=8)
proms_list = [
    x8(tensor([1, 2, 3], device=device)),
    x8(tensor([2, 3], device=device))
]
resp_list = [
    tensor([1, 2, 3], device=device),
    qnt.to(device)
]
out = model(txt_lst, proms_list, max_steps=200) # what is max_steps?
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

for i in range(100):
    optim.zero_grad()
    _ = model(tx_lst, promps_list, resp_list)
    losses = model.loss
    sum(losses.values()).backward()
    optim.step()

    if i % 20 == 0:
        print(f"iter={i}, {losses}.")

out = model(tx_list, proms_list, max_steps=200)

print(qnt), print(out)

codes = rearrange(out[1], "t -> 1 1 t")
wavs, sr = decode(codes)
```

(NAR) Components:
- init()
- forward()
- generate()

```python
resps      = _ # codes for 2-8?
num_quants = 1024

model = NAR(num_quants).to(device)

txt_list = [
    tensor([2, 3], device=device)
]

x8 = partial(repeat, pattern="t -> t 1", l=8)
proms_list = [
    x8(tensor([2, 3], device=device))
]
resps_x1_list = [
    resps[:1].t().device()
]
resps_x8_list = [
    resps.t().device()
]

codes = model(
    txt_list,
    proms_list,
    resps_list=resps_x1_list,
    sampling_temperature_0.2
)[0]

decode_to_file(codes, Path())

optim = torch.optim.Adam(model.parameters(), lr=1e-4)

for i in range(200):
    optim.zero_grad()

    _ = model(txt_list, proms_list, resps_list=resps_x8_list)

    losses = gather_attribute(model, "loss")
    loss = sum(losses.values())
    loss.backward()

    optim.step()

    if i % 20 == 0:
        stats = {k: v.item() for k, v in losses.items()}
        stats["loss"] = loss.item()
        print(f"iter={i}, {stats}.")
    
    for i in range(1, 8):
        resps_list = [
            resps[:i].t().to(device)
        ]

        codes = model(
            txt_list,
            proms_list,
            resps_list=resps_list,
            sampling_temperature=0.2
        )[0]

        decode_to_file(codes, Path())
```
"""

import math
import torch

from einops import rearrange
from torch import Tensor, einsum, nn


class SinusodialEmbedding(nn.Module):
    def __init__(self, d_model):
        """Initialize the SinusodialEmbedding module.
        
        Args:
            d_model (int): Dimension of the model.
        """
        super().__init__()
        self.d_model = d_model
        
        # Calculate the omega values for sinusoidal encoding.
        exponent = torch.arange(self.d_half, dtype=torch.float32)  # Create a tensor [0, 1, ..., d_half-1]
        exponent = exponent / self.d_half  # Normalize the exponent values.
        omega = torch.exp(-math.log(1e4) * exponent)  # Calculate the omega values.
        
        # Register the omega tensor as a buffer, without adding it to the module's parameters.
        self.register_buffer("omega", omega, persistent=False)

    @property
    def d_half(self):
        """Get half of the model's dimension, d_model. Only support even d_model values."""
        assert self.d_model % 2 == 0, "Only support even d_model."
        return self.d_model // 2

    def forward(self, x):
        """Compute the sinusoidal encoding for the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Sinusoidally encoded tensor with the same shape as input but extended in the last dimension.
        """
        omega = self.omega

        # Increase the dimensionality of omega tensor to match the dimensionality of input tensor x.
        while omega.dim() <= x.dim():
            omega = omega.unsqueeze(0)
            
        x = x.unsqueeze(-1)  # Increase the dimensionality of x by adding an extra dimension.
        x = omega * x  # Multiply omega with the extended x tensor.
        
        # Concatenate the sine and cosine values in the last dimension.
        x = torch.cat([x.sin(), x.cos()], dim=-1)

        return x
    
    def get_pe(self, n: int):
        """
        Args:
            n: int
        Returns:
            pe: (n d)
        """
        device = self.omega.device
        return self.forward(torch.arange(n, device=device))

    def add_pe(self, x):
        """
        Args:
            x: (b t c)
        """
        e = self.get_pe(x.shape[1])  # t d
        e = e[None]  # b t d
        x = x + e
        return x
    

# Lucidrains-style attention mechanism
class Attention(nn.Module):
    def __init__(self, d_model, n_heads, causal):
        super().__init__()
        assert d_model % n_heads == 0
        dim_head     = d_model // n_heads
        self.causal  = causal # AR vs NAR
        self.n_heads = n_heads
        self.scale   = dim_head**-0.5
        self.to_qkv  = nn.Linear(d_model, d_model * 3, bias=False)
        self.to_out  = nn.Linear(d_model, d_model)

    def forward(self, x, m):
        """
        Args:
            x: (b t c)
            m: (b t c), 1 is data, 0 is padding
        Returns:
            x: (b t c)
        """
        # num of attn heads
        h = self.n_heads

        # split tensor into 3 chunks (q, k, v)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split (head, dim_head) -> head, dim_head
        q, k, v = map(lambda t: rearrange(t, "b t (h d) -> b t h d", h=h), (q, k, v))

        # attn score := (b t h d * b t h d := b t t h)
        e = einsum("b i h d, b j h d -> b i j h", q, k)
        e = e * self.scale # scale result down for stability

        # (b t t 1) shape?
        kpm = m.unsqueeze(1) * m.unsqueeze(2)  # b i j 1

        # zero-out upper-right triangle for causal, only diff during forward pass?
        if self.causal:
            kpm = kpm.squeeze(-1).tril().unsqueeze(-1)  # b i j 1

        # fill zero, softmax remaining (normalise scores to 1.0 when summed)
        e = e.masked_fill(kpm == 0, -torch.finfo(e.dtype).max)
        a = e.softmax(dim=2)  # Normalize on j, i.e. key

        # final shape := (b t h d)
        o = einsum("b i j h, b j h d -> b i h d", a, v)
        o = o.flatten(-2)
        o = self.to_out(o)  # b t c

        o = o * m

        return o
    

class Base(nn.Module):
    def __init__(self):
        pass