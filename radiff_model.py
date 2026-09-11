import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens and diffusion steps."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch_size, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ConcatSquashLinear(nn.Module):
    """
    Conditioning Linear layer from RADiff.
    Combines input feature X with context embedding C and time step noise beta.
    """
    def __init__(self, in_features: int, out_features: int, context_dim: int):
        super().__init__()
        self._layer = nn.Linear(in_features, out_features)
        self._hyper_gate = nn.Linear(context_dim, out_features)
        self._hyper_bias = nn.Linear(context_dim, out_features, bias=False)

    def forward(self, ctx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        ctx: (batch_size, context_dim) or (batch_size, 1, context_dim)
        x: (batch_size, seq_len, in_features)
        """
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(1)
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        out = self._layer(x) * gate + bias
        return out

class SocialContextEncoder(nn.Module):
    """
    Context encoder combining observed past flight trajectory and RAG retrieved priors.
    """
    def __init__(self, obs_len: int = 12, pred_len: int = 24, context_dim: int = 256, in_channels: int = 3):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.obs_proj = nn.Linear(obs_len * in_channels, context_dim // 2)
        self.rag_proj = nn.Linear(pred_len * in_channels, context_dim // 2)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=context_dim, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out_proj = nn.Linear(context_dim, context_dim)

    def forward(self, obs_traj: torch.Tensor, rag_priors: torch.Tensor) -> torch.Tensor:
        """
        obs_traj: (B, obs_len, 3)
        rag_priors: (B, K, pred_len, 3)
        Returns: context embedding of shape (B, context_dim)
        """
        B = obs_traj.size(0)
        obs_flat = obs_traj.reshape(B, -1)                     # (B, obs_len * 3)
        obs_feat = self.obs_proj(obs_flat)                     # (B, context_dim // 2)
        
        # Aggregate RAG priors by averaging retrieved K neighbors
        rag_mean = torch.mean(rag_priors, dim=1)               # (B, pred_len, 3)
        rag_flat = rag_mean.reshape(B, -1)                     # (B, pred_len * 3)
        rag_feat = self.rag_proj(rag_flat)                     # (B, context_dim // 2)
        
        ctx_combined = torch.cat([obs_feat, rag_feat], dim=-1).unsqueeze(1) # (B, 1, context_dim)
        ctx_trans = self.transformer(ctx_combined).squeeze(1)               # (B, context_dim)
        return self.out_proj(ctx_trans)

class TransformerDenoisingModel(nn.Module):
    """
    Core Transformer Denoising Model from RADiff.
    Predicts added Gaussian noise given noisy target, diffusion timestep, and retrieved context.
    """
    def __init__(self, pred_len: int = 24, context_dim: int = 256, in_channels: int = 3, tf_layers: int = 3):
        super().__init__()
        self.pred_len = pred_len
        self.context_dim = context_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, context_dim // 2),
            nn.SiLU(),
            nn.Linear(context_dim // 2, context_dim // 2)
        )
        
        self.in_proj = nn.Linear(in_channels, context_dim)
        self.pos_encoder = PositionalEncoding(context_dim, max_len=pred_len + 10)
        
        self.concat1 = ConcatSquashLinear(context_dim, context_dim, context_dim + context_dim // 2)
        
        tf_layer = nn.TransformerEncoderLayer(d_model=context_dim, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(tf_layer, num_layers=tf_layers)
        
        self.concat2 = ConcatSquashLinear(context_dim, context_dim // 2, context_dim + context_dim // 2)
        self.out_proj = ConcatSquashLinear(context_dim // 2, in_channels, context_dim + context_dim // 2)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, pred_len, 3) noisy trajectory at step t
        t: (B, 1) normalized diffusion timestep [0, 1]
        context: (B, context_dim) encoded past & RAG priors
        Returns predicted noise epsilon of shape (B, pred_len, 3)
        """
        t_feat = self.time_embed(t)                            # (B, context_dim // 2)
        ctx_emb = torch.cat([context, t_feat], dim=-1)         # (B, context_dim + context_dim // 2)
        
        h = self.in_proj(x_t)                                  # (B, pred_len, context_dim)
        h = self.pos_encoder(h)
        h = self.concat1(ctx_emb, h)
        
        h_trans = self.transformer(h)                          # (B, pred_len, context_dim)
        h_mid = self.concat2(ctx_emb, h_trans)                 # (B, pred_len, context_dim // 2)
        eps_pred = self.out_proj(ctx_emb, h_mid)               # (B, pred_len, 3)
        return eps_pred

class RADiffTrajectoryGenerator(nn.Module):
    """
    Complete Retrieval-Augmented Trajectory Diffusion Model for NASA GUAM UAM trajectories.
    """
    def __init__(
        self,
        obs_len: int = 12,
        pred_len: int = 24,
        context_dim: int = 256,
        num_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_steps = num_steps
        
        self.context_encoder = SocialContextEncoder(obs_len, pred_len, context_dim)
        self.denoiser = TransformerDenoisingModel(pred_len, context_dim)
        
        # DDPM noise schedule parameters
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def forward_loss(self, obs_traj: torch.Tensor, pred_traj: torch.Tensor, rag_priors: torch.Tensor) -> torch.Tensor:
        """
        Computes DDPM training loss (MSE between actual noise and predicted noise).
        """
        B = obs_traj.size(0)
        device = obs_traj.device
        
        # Sample random diffusion timesteps for each batch sample
        t_indices = torch.randint(0, self.num_steps, (B,), device=device).long()
        t_norm = (t_indices.float() / self.num_steps).unsqueeze(1)  # (B, 1)
        
        noise = torch.randn_like(pred_traj)                         # (B, pred_len, 3)
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t_indices].view(B, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t_indices].view(B, 1, 1)
        
        # Add noise to target trajectory
        x_t = sqrt_alpha_cumprod * pred_traj + sqrt_one_minus_alpha * noise
        
        # Encode context & predict noise
        context = self.context_encoder(obs_traj, rag_priors)
        eps_pred = self.denoiser(x_t, t_norm, context)
        
        loss = F.mse_loss(eps_pred, noise)
        return loss

    @torch.no_grad()
    def generate(self, obs_traj: torch.Tensor, rag_priors: torch.Tensor) -> torch.Tensor:
        """
        Generates future UAM trajectories via reverse diffusion sampling.
        obs_traj: (B, obs_len, 3)
        rag_priors: (B, K, pred_len, 3)
        Returns predicted future trajectory: (B, pred_len, 3)
        """
        B = obs_traj.size(0)
        device = obs_traj.device
        context = self.context_encoder(obs_traj, rag_priors)
        
        # Start from standard Gaussian noise
        x_t = torch.randn(B, self.pred_len, 3, device=device)
        
        for t_step in reversed(range(self.num_steps)):
            t_tensor = torch.full((B, 1), t_step / self.num_steps, device=device, dtype=torch.float)
            eps_pred = self.denoiser(x_t, t_tensor, context)
            
            beta_t = self.betas[t_step]
            alpha_t = self.alphas[t_step]
            alpha_cumprod_t = self.alphas_cumprod[t_step]
            
            if t_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0.0
                
            sigma_t = torch.sqrt(beta_t)
            mean_t = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * eps_pred)
            x_t = mean_t + sigma_t * noise
            
        return x_t

if __name__ == '__main__':
    model = RADiffTrajectoryGenerator(obs_len=12, pred_len=24, context_dim=128, num_steps=20)
    obs = torch.randn(4, 12, 3)
    pred = torch.randn(4, 24, 3)
    priors = torch.randn(4, 3, 24, 3)
    
    loss = model.forward_loss(obs, pred, priors)
    print("Training MSE Loss:", loss.item())
    
    generated = model.generate(obs, priors)
    print("Generated trajectory shape:", generated.shape)
