"""
Contains adaptations of existing VALL-E implementation from:
https://github.com/enhuiz/vall-e

(Base) Components:
- init()
- [x] SinusodialEmbedding (2017 original positional embedding)
- [x] Attention (Single head attention)
- [x] AdaLN (Adaptive Layer Norm, better for generative networks)
- [x] PrenormResidual (Norm before attn/ffwd), opposite of 2017 paper
- [x] Block (TransformerBlock)
- [ ] Embedding (Generic forward pass for an embedding)
- [ ] MultiEmbedding (Sum embeddings on different levels)

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
from torch.utils.checkpoint import checkpoint

# AdaLN used for AR, Standard LN used for NAR?
class PrenormResidual(nn.Module):
    def __init__(
        self,
        block,
        d_model,
        p_dropout,
        requires_mask=False,
        norm_type="ln",
        n_levels: int = None,
    ):
        super().__init__()
        self.block = block
        self.requires_mask = requires_mask
        self.norm_type = norm_type
        if norm_type == "ln":
            self.norm = nn.LayerNorm(d_model)
        elif norm_type == "adaln":
            assert n_levels is not None
            self.norm = AdaLN(d_model, n_levels)
        else:
            raise NotImplementedError(norm_type)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, m, l):
        """
        Args:
            x: input (b t d)
            m: mask (b t 1), 1 is valuable and 0 is padding
            l: level to use, required only for AdaLN
        """
        nopts = {"l": l} if self.norm_type == "adaln" else {}
        bopts = {"m": m} if self.requires_mask else {}
        x = x + self.dropout(self.block(self.norm(x, **nopts) * m, **bopts))
        return x * m


# Adaptive layer norm (Better for generative networks due to robustness against
# highly varying data?)
class AdaLN(nn.Module):
    def __init__(self, d_model, n_levels, eps=1e-5, k=0.1, c=2):
        super().__init__()
        self.eps = eps
        self.emb = nn.Embedding(n_levels, d_model * 2)
        self.k   = k
        self.c   = c
        nn.init.zeros_(self.emb.weight)
    
    def forward(self, x, l):
        logy, β = self.emb(l).unsqueeze(1).chunk(2, dim=-1)
        h = F.layer_norm(x, x.shape[-1:], eps=self.eps)

        h = self.c * (1 - (self.k * h).detach()) * h
        y = logy.exp() * h + β

        return y
    

# Standard sine position embedding
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
    

class Block(nn.Sequential):
    def __init__(self, d_model, n_heads, p_dropout, casual, norm_type, n_levels):
        super().__init__()
        self.attn = PrenormResidual(
            Attention(d_model, n_heads, casual),
            d_model=d_model,
            p_dropout=p_dropout,
            # (for causal: left-to-right)
            # (for causal + non-causal: masks padding)
            requires_mask=True, 
            norm_type=norm_type,
            n_levels=n_levels,
        )
        self.ffn = PrenormResidual(
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(p_dropout),
                nn.Linear(d_model * 4, d_model),
            ),
            d_model=d_model,
            p_dropout=p_dropout,
            norm_type=norm_type,
            n_levels=n_levels,
        )

    def forward(self, x, m, l):
        """
        Args:
            x: (b t c)
            m: (b t 1)
            l: (b)
        """
        poor_in_vram = True
        if x.requires_grad and poor_in_vram:
            # NOTE: # Re-compute forward pass
            # Trade compute for memory, but also keeps RNG consistent?
            x = checkpoint(self.attn, x, m, l) 
        else:
            x = self.attn(x, m, l)
        x = self.ffn(x, m, l)
        return x
    
class Base(nn.Module):
    def __init__(self):
        pass