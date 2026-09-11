import torch
import torch.nn as nn
from torch.nn import Module

from models.diffusion.layers import ConcatSquashLinear, PositionalEncoding


class st_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        channel_in = 3
        channel_out = 32
        dim_kernel = 3
        self.dim_embedding_key = 256
        self.spatial_conv = nn.Conv1d(
            channel_in, channel_out, dim_kernel, stride=1, padding=1
        )
        self.temporal_encoder = nn.GRU(
            channel_out, self.dim_embedding_key, 1, batch_first=True
        )
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.spatial_conv.weight)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
        nn.init.zeros_(self.spatial_conv.bias)
        nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
        nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

    def forward(self, x):
        x_t = torch.transpose(x, 1, 2)
        x_after_spatial = self.relu(self.spatial_conv(x_t))
        x_embed = torch.transpose(x_after_spatial, 1, 2)
        _, state_x = self.temporal_encoder(x_embed)
        return state_x.squeeze(0)


class social_transformer(nn.Module):
    def __init__(self, obs=11):
        super().__init__()
        self.obs = obs
        self.encode_past = nn.Linear(3 * self.obs, 256, bias=False)
        self.layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=2, dim_feedforward=256
        )
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

    def forward(self, h, mask):
        h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1)
        h_feat_ = self.transformer_encoder(h_feat)
        return h_feat + h_feat_


class TransformerDenoisingModel(Module):
    def __init__(self, context_dim=256, tf_layer=2, obs=11):
        super().__init__()
        self.encoder_context = social_transformer(obs=obs)
        self.pos_emb = PositionalEncoding(
            d_model=2 * context_dim, dropout=0.1, max_len=24
        )
        self.concat1 = ConcatSquashLinear(3, 2 * context_dim, context_dim + 3)
        self.layer = nn.TransformerEncoderLayer(
            d_model=2 * context_dim,
            nhead=2,
            dim_feedforward=2 * context_dim,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.layer, num_layers=tf_layer
        )
        self.concat3 = ConcatSquashLinear(2 * context_dim, context_dim, context_dim + 3)
        self.concat4 = ConcatSquashLinear(
            context_dim, context_dim // 2, context_dim + 3
        )
        self.linear = ConcatSquashLinear(context_dim // 2, 3, context_dim + 3)

        self.transformer_decoder = nn.TransformerDecoder(
            self.layer, num_layers=tf_layer
        )
        self.layer_decoder = nn.TransformerDecoderLayer(
            d_model=2 * context_dim,
            nhead=2,
            dim_feedforward=2 * context_dim,
        )

    def forward(self, x, beta, context, mask):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        context = self.encoder_context(context, mask)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        ctx_emb = torch.cat([time_emb, context], dim=-1)
        x = self.concat1(ctx_emb, x)
        final_emb = self.pos_emb(x.permute(1, 0, 2))

        trans = self.transformer_encoder(final_emb).permute(1, 0, 2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)

    def generate_accelerate(self, x, beta, context, mask):
        sample_num = x.shape[1]
        points_num = x.shape[2]
        beta = beta.view(beta.size(0), 1, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        context = self.encoder_context(context, mask)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        ctx_emb = torch.cat([time_emb, context], dim=-1)
        ctx_emb = ctx_emb.repeat(1, sample_num, 1).unsqueeze(2)

        x = self.concat1.batch_generate(ctx_emb, x)
        x = x.contiguous().view(-1, points_num, 512)
        final_emb = self.pos_emb(x.permute(1, 0, 2))

        trans = self.transformer_encoder(final_emb).permute(1, 0, 2)
        trans = trans.contiguous().view(-1, sample_num, points_num, 512)
        trans = self.concat3.batch_generate(ctx_emb, trans)
        trans = self.concat4.batch_generate(ctx_emb, trans)
        return self.linear.batch_generate(ctx_emb, trans)
