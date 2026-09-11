import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):

    half = dim // 2

    freqs = torch.exp(
        -torch.arange(half, device=t.device).float()
        * (torch.log(torch.tensor(max_period)) / half)
    )

    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)

    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)

    if dim % 2:

        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)

    return emb


class AdaLN(nn.Module):

    def __init__(self, d_model, cond_dim):

        super().__init__()

        self.to_scale_shift = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * d_model))

    def forward(self, x, cond):

        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)

        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerBlock(nn.Module):

    def __init__(self, d_model=256, n_heads=8, mlp_ratio=4.0, cond_dim=256):

        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.adaln1 = AdaLN(d_model, cond_dim)

        self.ln2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

        self.adaln2 = AdaLN(d_model, cond_dim)

    def forward(self, x, cond):

        h = self.ln1(x)

        h = self.adaln1(h, cond)

        h, _ = self.attn(h, h, h, need_weights=False)

        x = x + h

        h = self.ln2(x)

        h = self.adaln2(h, cond)

        h = self.mlp(h)

        x = x + h

        return x


class TrajDiT(nn.Module):

    def __init__(self, T_pred=30, in_dim=3, d_model=256, n_layers=6, cond_dim=128):

        super().__init__()

        self.T_pred, self.in_dim = T_pred, in_dim

        self.token_in = nn.Linear(in_dim, d_model)

        self.pos = nn.Parameter(torch.zeros(1, T_pred, d_model))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model, n_heads=8, mlp_ratio=4, cond_dim=cond_dim
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_out = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, in_dim)

    def forward(self, x, t, cond_vec):

        B = 1

        x = x.unsqueeze(0)

        h = self.token_in(x) + self.pos

        t_emb = timestep_embedding(t.unsqueeze(0), h.size(-1))

        cond = cond_vec.unsqueeze(0) + t_emb

        for blk in self.blocks:

            h = blk(h, cond)

        h = self.ln_out(h)

        eps = self.head(h)

        return eps.squeeze(0)
