import time

from torch import nn

from typing import Optional, Union, Tuple

from models.tcn_model import TemporalConvNet

from models.batch_gat import GAT

from models.cvae_base import CVAE

from models.utils import acc_to_abs, DotDict, endpoint_smoothing_torch

import numpy as np

import matplotlib.pyplot as plt

import torch.nn.functional as F

from torch_kmeans import KMeans


from models.diffusion.model_diffusion import (
    TransformerDenoisingModel as CoreDenoisingModel,
)


from models.diffusion.layers import MLP, social_transformer, st_encoder

from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math


import torch


NUM_Tau = 5


class TrajAirNet(nn.Module):

    def __init__(self, args):

        super(TrajAirNet, self).__init__()

        input_size = args.input_channels

        n_classes = int(args.preds / args.preds_step)

        num_channels = [args.tcn_channel_size] * args.tcn_layers

        num_channels.append(n_classes)

        tcn_kernel_size = args.tcn_kernels

        dropout = args.dropout

        graph_hidden = args.graph_hidden

        gat_in = n_classes * args.obs

        gat_out = n_classes * args.obs

        alpha = args.alpha

        cvae_encoder = [n_classes * n_classes]

        for layer in range(args.cvae_layers):

            cvae_encoder.append(args.cvae_channel_size)

        cvae_decoder = [args.cvae_channel_size] * args.cvae_layers

        cvae_decoder.append(input_size * args.mlp_layer)

        self.tcn_encoder_x = TemporalConvNet(
            input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout
        )

        self.tcn_encoder_similarest = TemporalConvNet(
            input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout
        )

        self.tcn_encoder_y = TemporalConvNet(
            input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout
        )

        self.tcn_encoder_search = TemporalConvNet(
            input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout
        )

        self.cvae = CVAE(
            encoder_layer_sizes=cvae_encoder,
            latent_size=args.cvae_hidden,
            decoder_layer_sizes=cvae_decoder,
            conditional=True,
            num_labels=gat_out + gat_in,
        )

        self.gat = GAT(
            in_feature=gat_in,
            hidden_feature=graph_hidden,
            out_feature=gat_out,
            attention_layers=3,
            dropout=0.1,
            alpha=alpha,
        )

        self.linear_decoder = nn.Linear(args.mlp_layer, n_classes)

        self.context_conv = nn.Conv1d(
            in_channels=5, out_channels=4, kernel_size=args.cnn_kernels
        )

        self.context_linear = nn.Linear(11, args.num_context_output_c)

        self.relu = nn.ReLU()

        self.init_weights()

        self.model = CoreDenoisingModel().cuda()

        self.betas = self.make_beta_schedule(
            schedule="linear", n_timesteps=100, start=1.0e-4, end=5.0e-2
        ).cuda()

        self.alphas = 1 - self.betas

        self.alphas_prod = torch.cumprod(self.alphas, 0)

        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)

        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        self.social_encoder = social_transformer(args.obs)

        n_clusters = 3

        self.kmeans = KMeans(n_clusters=n_clusters)

        embed_dim = 128

        encoder_layers = TransformerEncoderLayer(embed_dim, 8, embed_dim, 0.15)

        self.transformer_encoder = TransformerEncoder(encoder_layers, 4)

        self.enc_embed = EncTokenEmbedding(3, embed_dim)

        self.pos_emb = PositionalEncoding(embed_dim, dropout=0.1, max_len=11)

        self.cross_att_0 = CrossAttnBlock(
            d_model=embed_dim, n_heads=16, seq_length=12, dropout=dropout
        )

        self.cross_att_1 = CrossAttnBlock(
            d_model=embed_dim, n_heads=16, seq_length=12, dropout=dropout
        )

        self.cross_att_2 = CrossAttnBlock(
            d_model=embed_dim, n_heads=16, seq_length=12, dropout=dropout
        )

        self.norm_attn = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * args.obs, 512), nn.ReLU(), nn.Linear(512, 256)
        )

        self.mlp_centers = nn.Sequential(
            nn.Linear(n_clusters * 3, 64),
            nn.ReLU(),
        )

        self.centers_proj = nn.Linear(3, 128)

        self.obs_proj = nn.Linear(3, embed_dim)

        self.linear_reg = nn.Linear(embed_dim, 3)

    def init_weights(self):

        self.linear_decoder.weight.data.normal_(0, 0.05)

        self.context_linear.weight.data.normal_(0, 0.05)

        self.context_conv.weight.data.normal_(0, 0.1)

    def calculate_mask(self, obs_traj):

        relation_dist = 8

        last_points = obs_traj[:, -1, :]

        diff = last_points[:, None, :] - last_points[None, :, :]

        dist_matrix = diff.norm(p=2, dim=-1)

        adj = F.softmax(-dist_matrix / 1, dim=0)

        mask = dist_matrix < relation_dist

        return adj, mask

    def forward(
        self,
        x,
        y,
        adj,
        context,
        obs_traj_search_results,
        pred_traj_search_results,
        sort=False,
    ):

        batch_size = x.shape[0]

        agent_num = x.shape[1]

        var_num = x.shape[2]

        ob_len = x.shape[3]

        topk = obs_traj_search_results.shape[2]

        pre_len = obs_traj_search_results.shape[3]

        obs_traj_search_results = torch.FloatTensor(obs_traj_search_results).to(
            x.device
        )

        searched_for_cluster = obs_traj_search_results.clone()

        end_points = torch.reshape(
            searched_for_cluster, (batch_size * agent_num, topk, pre_len, var_num)
        )

        end_points = end_points[:, :, -1, :]

        centers = self.kmeans(end_points).centers

        centers_embed = self.centers_proj(centers)

        centers_embed_list = torch.split(centers_embed, 1, dim=1)

        x1 = torch.reshape(x, (batch_size * agent_num, var_num, ob_len))

        x1_proj = self.obs_proj(x1.permute(0, 2, 1))

        b, c, h = x1_proj.shape

        cross_weight_0 = self.cross_att_0(centers_embed_list[0], x1_proj).view(b, c, h)

        cross_weight_1 = self.cross_att_1(centers_embed_list[1], x1_proj).view(b, c, h)

        cross_weight_2 = self.cross_att_2(centers_embed_list[2], x1_proj).view(b, c, h)

        cross_feature = cross_weight_0 + cross_weight_1 + cross_weight_2

        cross_feature = self.linear_reg(cross_feature)

        encoded_x = self.tcn_encoder_x(x1)

        encoded_x = torch.flatten(encoded_x, start_dim=1, end_dim=-1)

        encoded_x = torch.reshape(
            encoded_x, (batch_size, agent_num, encoded_x.shape[1])
        )

        serached1 = torch.reshape(
            obs_traj_search_results, (batch_size * agent_num * topk, pre_len, var_num)
        )

        serached1 = torch.transpose(serached1, 1, 2)

        encoded_search = self.tcn_encoder_search(serached1)

        encoded_search = torch.reshape(
            encoded_search,
            (
                batch_size,
                agent_num,
                topk,
                encoded_search.shape[1],
                encoded_search.shape[2],
            ),
        )

        encoded_search = torch.flatten(encoded_search, start_dim=3, end_dim=-1)

        _traj_mask = torch.ones(agent_num, agent_num).cuda()

        gat_out = self.gat(encoded_x, _traj_mask)

        H_x = torch.repeat_interleave(gat_out.unsqueeze(2), topk, dim=2)

        H_xx = torch.repeat_interleave(encoded_x.unsqueeze(2), topk, dim=2)

        H_x = torch.cat((H_xx, H_x), dim=-1)

        H_x = torch.reshape(H_x, ((batch_size * agent_num), H_x.shape[2], H_x.shape[3]))

        H_y = torch.reshape(
            encoded_search,
            (
                (batch_size * agent_num),
                encoded_search.shape[2],
                encoded_search.shape[3],
            ),
        )

        H_yy, means, log_var, z = self.cvae(H_y, H_x)

        H_yy = torch.reshape(H_yy, (H_yy.shape[0], H_yy.shape[1], 3, -1))

        recon_y_x = (self.linear_decoder(H_yy)).permute(0, 1, 3, 2)

        fut_traj = torch.reshape(y, (batch_size * agent_num, y.shape[2], y.shape[3]))

        fut_traj = fut_traj.permute(0, 2, 1)

        past_traj = torch.reshape(x, (batch_size * agent_num, x.shape[2], x.shape[3]))

        past_traj = past_traj.permute(0, 2, 1)

        traj_mask = torch.zeros(batch_size * agent_num, batch_size * agent_num).cuda()

        for i in range(batch_size):

            traj_mask[
                i * agent_num : (i + 1) * agent_num, i * agent_num : (i + 1) * agent_num
            ] = 1.0

        generated_y = self.p_sample_loop_accelerate(cross_feature, traj_mask, recon_y_x)

        loss_dist = (
            (generated_y - fut_traj.unsqueeze(dim=1))
            .norm(p=2, dim=-1)
            .mean(dim=-1)
            .min(dim=1)[0]
            .mean()
        )

        loss_KL = -0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp())

        loss_uncertainty = (
            (generated_y - fut_traj.unsqueeze(dim=1))
            .norm(p=2, dim=-1)
            .mean(dim=(1, 2))
            .mean()
        )

        return loss_dist, loss_uncertainty, loss_KL

    def inference(
        self,
        x,
        y,
        adj,
        context,
        obs_traj_search_results,
        pred_traj_search_results,
    ):

        batch_size = x.shape[0]

        agent_num = x.shape[1]

        var_num = x.shape[2]

        ob_len = x.shape[3]

        topk = obs_traj_search_results.shape[2]

        pre_len = obs_traj_search_results.shape[3]

        obs_traj_search_results = torch.FloatTensor(obs_traj_search_results).to(
            x.device
        )

        searched_for_cluster = obs_traj_search_results.clone()

        end_points = torch.reshape(
            searched_for_cluster, (batch_size * agent_num, topk, pre_len, var_num)
        )

        end_points = end_points[:, :, -1, :]

        centers = self.kmeans(end_points).centers

        centers_embed = self.centers_proj(centers)

        centers_embed_list = torch.split(centers_embed, 1, dim=1)

        x1 = torch.reshape(x, (batch_size * agent_num, var_num, ob_len))

        x1_proj = self.obs_proj(x1.permute(0, 2, 1))

        b, c, h = x1_proj.shape

        cross_weight_0 = self.cross_att_0(centers_embed_list[0], x1_proj).view(b, c, h)

        cross_weight_1 = self.cross_att_1(centers_embed_list[1], x1_proj).view(b, c, h)

        cross_weight_2 = self.cross_att_2(centers_embed_list[2], x1_proj).view(b, c, h)

        cross_feature = cross_weight_0 + cross_weight_1 + cross_weight_2

        cross_feature = self.linear_reg(cross_feature)

        encoded_x = self.tcn_encoder_x(x1)

        encoded_x = torch.flatten(encoded_x, start_dim=1, end_dim=-1)

        encoded_x = torch.reshape(
            encoded_x, (batch_size, agent_num, encoded_x.shape[1])
        )

        serached1 = torch.reshape(
            obs_traj_search_results, (batch_size * agent_num * topk, pre_len, var_num)
        )

        serached1 = torch.transpose(serached1, 1, 2)

        encoded_search = self.tcn_encoder_search(serached1)

        encoded_search = torch.reshape(
            encoded_search,
            (
                batch_size,
                agent_num,
                topk,
                encoded_search.shape[1],
                encoded_search.shape[2],
            ),
        )

        encoded_search = torch.flatten(encoded_search, start_dim=3, end_dim=-1)

        _traj_mask = torch.ones(agent_num, agent_num).cuda()

        gat_out = self.gat(encoded_x, _traj_mask)

        H_x = torch.repeat_interleave(gat_out.unsqueeze(2), topk, dim=2)

        H_xx = torch.repeat_interleave(encoded_x.unsqueeze(2), topk, dim=2)

        H_x = torch.cat((H_xx, H_x), dim=-1)

        H_x = torch.reshape(H_x, ((batch_size * agent_num), H_x.shape[2], H_x.shape[3]))

        H_y = torch.reshape(
            encoded_search,
            (
                (batch_size * agent_num),
                encoded_search.shape[2],
                encoded_search.shape[3],
            ),
        )

        H_yy, means, log_var, z = self.cvae(H_y, H_x)

        H_yy = torch.reshape(H_yy, (H_yy.shape[0], H_yy.shape[1], 3, -1))

        recon_y_x = (self.linear_decoder(H_yy)).permute(0, 1, 3, 2)

        fut_traj = torch.reshape(y, (batch_size * agent_num, y.shape[2], y.shape[3]))

        fut_traj = fut_traj.permute(0, 2, 1)

        past_traj = torch.reshape(x, (batch_size * agent_num, x.shape[2], x.shape[3]))

        past_traj = past_traj.permute(0, 2, 1)

        traj_mask = torch.zeros(batch_size * agent_num, batch_size * agent_num).cuda()

        for i in range(batch_size):

            traj_mask[
                i * agent_num : (i + 1) * agent_num, i * agent_num : (i + 1) * agent_num
            ] = 1.0

        generated_y = self.p_sample_loop_accelerate(cross_feature, traj_mask, recon_y_x)

        generated_y = torch.reshape(
            generated_y,
            (
                generated_y.shape[0] * generated_y.shape[1],
                generated_y.shape[2],
                generated_y.shape[3],
            ),
        )

        smoothed = endpoint_smoothing_torch(generated_y, lam=1.0)

        generated_y = torch.reshape(
            smoothed,
            (batch_size * agent_num, topk, smoothed.shape[1], smoothed.shape[2]),
        )

        return generated_y

    def extract(self, input, t, x):

        shape = x.shape

        out = torch.gather(input, 0, t.to(input.device))

        reshape = [t.shape[0]] + [1] * (len(shape) - 1)

        return out.reshape(*reshape)

    def make_beta_schedule(
        self,
        schedule: str = "linear",
        n_timesteps: int = 1000,
        start: float = 1e-5,
        end: float = 1e-2,
    ) -> torch.Tensor:
        """
        Make beta schedule.

        Parameters
        ----
        schedule: str, in ['linear', 'quad', 'sigmoid'],
        n_timesteps: int, diffusion steps,
        start: float, beta start, `start<end`,
        end: float, beta end,

        Returns
        ----
        betas: Tensor with the shape of (n_timesteps)

        """

        if schedule == "linear":

            betas = torch.linspace(start, end, n_timesteps)

        elif schedule == "quad":

            betas = torch.linspace(start**0.5, end**0.5, n_timesteps) ** 2

        elif schedule == "sigmoid":

            betas = torch.linspace(-6, 6, n_timesteps)

            betas = torch.sigmoid(betas) * (end - start) + start

        return betas

    def noise_estimation_loss(self, x, y_0, mask):

        batch_size = x.shape[0]

        t = torch.randint(0, self.n_steps, size=(batch_size // 2 + 1,)).to(x.device)

        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:batch_size]

        a = self.extract(self.alphas_bar_sqrt, t, y_0)

        beta = self.extract(self.betas, t, y_0)

        am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, y_0)

        e = torch.randn_like(y_0)

        y = y_0 * a + e * am1

        output = self.model(y, beta, x, mask)

        return (e - output).square().mean()

    def p_sample(self, x, mask, cur_y, t):

        if t == 0:

            z = torch.zeros_like(cur_y).to(x.device)

        else:

            z = torch.randn_like(cur_y).to(x.device)

        t = torch.tensor([t]).cuda()

        eps_factor = (1 - self.extract(self.alphas, t, cur_y)) / self.extract(
            self.one_minus_alphas_bar_sqrt, t, cur_y
        )

        beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)

        eps_theta = self.model(cur_y, beta, x, mask)

        mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (
            cur_y - (eps_factor * eps_theta)
        )

        z = torch.randn_like(cur_y).to(x.device)

        sigma_t = self.extract(self.betas, t, cur_y).sqrt()

        sample = mean + sigma_t * z

        return sample

    def p_sample_accelerate(self, x, mask, cur_y, t):

        if t == 0:

            z = torch.zeros_like(cur_y).to(x.device)

        else:

            z = torch.randn_like(cur_y).to(x.device)

        t = torch.tensor([t]).cuda()

        eps_factor = (1 - self.extract(self.alphas, t, cur_y)) / self.extract(
            self.one_minus_alphas_bar_sqrt, t, cur_y
        )

        beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)

        eps_theta = self.model.generate_accelerate(cur_y, beta, x, mask)

        mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (
            cur_y - (eps_factor * eps_theta)
        )

        z = torch.randn_like(cur_y).to(x.device)

        sigma_t = self.extract(self.betas, t, cur_y).sqrt()

        sample = mean + sigma_t * z * 0.00001

        return sample

    def p_sample_loop(self, x, mask, shape):

        self.model.eval()

        prediction_total = torch.Tensor().cuda()

        for _ in range(20):

            cur_y = torch.randn(shape).to(x.device)

            for i in reversed(range(self.n_steps)):

                cur_y = self.p_sample(x, mask, cur_y, i)

            prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)

        return prediction_total

    def p_sample_loop_mean(self, x, mask, loc):

        prediction_total = torch.Tensor().cuda()

        for loc_i in range(1):

            cur_y = loc

            for i in reversed(range(NUM_Tau)):

                cur_y = self.p_sample(x, mask, cur_y, i)

            prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)

        return prediction_total

    def p_sample_loop_accelerate(self, x, mask, loc):
        """
        Batch operation to accelerate the denoising process.
        """

        split_num = loc.shape[1] // 2

        prediction_total = torch.Tensor().cuda()

        cur_y = loc[:, :split_num]

        for i in reversed(range(NUM_Tau)):

            cur_y = self.p_sample_accelerate(x, mask, cur_y, i)

        cur_y_ = loc[:, split_num:]

        for i in reversed(range(NUM_Tau)):

            cur_y_ = self.p_sample_accelerate(x, mask, cur_y_, i)

        prediction_total = torch.cat((cur_y_, cur_y), dim=1)

        return prediction_total


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):

        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[: x.size(0), :]

        return self.dropout(x)


class EncTokenEmbedding(nn.Module):

    def __init__(self, c_in, d_model):

        super(EncTokenEmbedding, self).__init__()

        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )

        for m in self.modules():

            if isinstance(m, nn.Conv1d):

                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):

        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)

        return x


class CrossAttnBlock(torch.nn.Module):
    r"""
    Single Block for Cross Attention

    Args:
        m1: first modality
        m2: second modality

    Shapes:
        m1: (seq_length, N_samples, N_features)
        m2: (seq_length, N_samples, N_features)

    Returns:
        embedding of m1 depending on attending on certain elements of m2, multihead_attn(k_m1, v_m1, q_m2)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        seq_length: int,
        d_k=None,
        d_v=None,
        qkv_bias=True,
        add_positional: Optional[bool] = False,
    ):

        super(CrossAttnBlock, self).__init__()

        self.d_model = d_model

        d_k = d_model // n_heads if d_k is None else d_k

        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.add_positional = add_positional

        if self.add_positional:

            self.positional_encoding = PositionalEncoding(d_model, dropout, seq_length)

        self._to_key = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)

        self._to_query = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)

        self._to_value = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)

        self.attn = torch.nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout
        )

    def forward(
        self,
        m1_x: Union[
            torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
        m2_x: Union[
            torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self.add_positional:

            m1_x = self.positional_encoding(m1_x)

            m2_x = self.positional_encoding(m2_x)

        m1_x = m2_x.view(-1, self.d_model)

        m2_x = m2_x.view(-1, self.d_model)

        m1_k = self._to_key(m1_x)

        m1_v = self._to_query(m1_x)

        m2_q = self._to_value(m2_x)

        cross_x, attn_weights = self.attn(m1_k, m1_v, m2_q)

        return cross_x


class LayerNorm(nn.Module):
    """
    Layer normalization.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:

        super(LayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))

        self.bias = nn.Parameter(torch.zeros(hidden_size))

        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.size(-1) == 1:

            return x

        u = x.mean(-1, keepdim=True)

        s = (x - u).pow(2).mean(-1, keepdim=True)

        x = (x - u) / torch.sqrt(s + self.variance_epsilon)

        return self.weight * x + self.bias


class MLP(nn.Module):

    def __init__(self, hidden_size: int, out_features: Optional[int] = None) -> None:

        super(MLP, self).__init__()

        if out_features is None:

            out_features = hidden_size

        self.linear = nn.Linear(hidden_size, out_features)

        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        hidden_states = self.linear(hidden_states)

        hidden_states = self.layer_norm(hidden_states)

        hidden_states = F.relu(hidden_states)

        return hidden_states
