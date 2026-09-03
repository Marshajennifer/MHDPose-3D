## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from math import sqrt

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from mamba_ssm import Mamba
from common.kinematic_rope import apply_rope_kinematic, build_phi_bias_from_parents
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., changedim=False, currentdim=0, depth=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., rope_eps=0.01, rope_base=10000.0, comb=False, vis=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) 

        self.proj_drop = nn.Dropout(proj_drop)

        self.rope_eps = float(rope_eps)
        self.rope_base = float(rope_base)
        self.register_buffer("pos_ids", None, persistent=True)
        self.register_buffer("phi_bias", None, persistent=True)

        self.comb = comb
        self.vis = vis

    @torch.no_grad()
    def set_kinematic_rope(
        self,
        num_joints,
        parents_np,
        pos_ids=None,
        joints_left=None,
        joints_right=None,
        depth_weight=0.5,
        side_weight=0.1,
    ):
        if pos_ids is None:
            pos_ids = torch.arange(num_joints, dtype=torch.long)
        self.pos_ids = pos_ids
        self.phi_bias = build_phi_bias_from_parents(
            parents_np,
            joints_left=joints_left,
            joints_right=joints_right,
            depth_weight=depth_weight,
            side_weight=side_weight,
        ).to(torch.float32)

    def forward(self, x, vis=False):
        B, N, C = x.shape
        H = self.num_heads
        Dh =  self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, H, Dh).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (B,H,N,Dh)

        if (self.pos_ids is not None) and (self.phi_bias is not None) and (self.rope_eps != 0.0):
            q, k = apply_rope_kinematic(
                q, k, self.pos_ids, self.phi_bias,
                base=self.rope_base, eps=self.rope_eps
            )

# token attention → SDPA
        # F.scaled_dot_product_attention picks Flash/Math/Memory-efficient automatically
        #If you pass scale=None → PyTorch automatically applies head_dim ** -0.5
        # If you pass your own scale → that value overrides the default.
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=(self.attn_drop.p if self.training else 0.0), scale=self.scale)
        x = x.transpose(1,2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, changedim=False, currentdim=0, depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, vis=vis)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        if self.changedim and self.currentdim < self.depth//2:
            self.reduction = nn.Conv1d(dim, dim//2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth//2:
            self.improve = nn.Conv1d(dim, dim*2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.vis = vis

    def forward(self, x, vis=False):
        x = x + self.drop_path(self.attn(self.norm1(x), vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if self.changedim and self.currentdim < self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TemporalMambaCore(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
    def forward(self, x):  # (B*, F, C)
        return self.mamba(self.norm(x))  # NO residual here

class BiTemporalMambaBlockV2(nn.Module):
    """Bidirectional temporal block with single residual."""
    def __init__(self, dim, mlp_ratio=2.0, drop=0.0, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.fwd  = TemporalMambaCore(dim, d_state, d_conv, expand)
        self.bwd  = TemporalMambaCore(dim, d_state, d_conv, expand)
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.Dropout(drop),
        )
        hid = int(dim * mlp_ratio)
        self.mix  = nn.Sequential(
            nn.Linear(dim, hid), nn.SiLU(), nn.Dropout(drop),
            nn.Linear(hid, dim), nn.Dropout(drop),
        )
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, x):  # (B*, F, C)
        y_f = self.fwd(x)
        x_rev = torch.flip(x, dims=[1]).contiguous()
        y_b = torch.flip(self.bwd(x_rev), dims=[1]).contiguous()
        y = self.fuse(torch.cat([y_f, y_b], dim=-1))
        g = self.gate(torch.cat([x, y], dim=-1))
        return x + g * self.mix(y)


# Residual SSM head (post-decoder temporal repair)
class ResidualSSMHead(nn.Module):
    def __init__(self, coord_dim=3, d_model=64, mlp_ratio=2.0, drop=0.0,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        hid = int(d_model * mlp_ratio)
        self.inp = nn.Sequential(
            nn.LayerNorm(coord_dim),
            nn.Linear(coord_dim, d_model),
        )
        self.core = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, coord_dim),
        )
        self.drop = nn.Dropout(drop)

    def _run(self, y):  # (B*, F, 3)
        y0 = y
        y = self.inp(y)
        y = self.core(y)
        y = self.out(y)
        return y0 + self.drop(y)

    def forward(self, x):
        if x.dim() == 4:  # (B,F,N,3)
            B, F, N, C = x.shape
            y = x.permute(0, 2, 1, 3).contiguous().view(B * N, F, C)
            y = self._run(y).view(B, N, F, C).permute(0, 2, 1, 3).contiguous()
            return y
        elif x.dim() == 5:  # (B,H,F,N,3)
            B, H, F, N, C = x.shape
            y = x.permute(0, 1, 3, 2, 4).contiguous().view(B * H * N, F, C)
            y = self._run(y).view(B, H, N, F, C).permute(0, 1, 3, 2, 4).contiguous()
            return y
        else:
            raise ValueError(f"ResidualSSMHead expects 4D or 5D, got {x.shape}")

class PMamba(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, d_model_temp=128, d_state=16, d_conv=4, expand=2,
                 residual_d_model=64, spatial_mlp_ratio=None, temporal_mlp_ratio=None, ste_stride=1,
                 norm_layer=None, is_train=True):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        spatial_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        temporal_dim = d_model_temp
        out_dim = 3     #### output dimension is num_joints * 3
        self.is_train=is_train
        self.ste_stride = max(1, int(ste_stride))
        spatial_mlp_ratio = mlp_ratio if spatial_mlp_ratio is None else spatial_mlp_ratio
        temporal_mlp_ratio = mlp_ratio if temporal_mlp_ratio is None else temporal_mlp_ratio

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans + 3, spatial_dim)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, spatial_dim))


        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, temporal_dim))


        self.pos_drop = nn.Dropout(p=drop_rate)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(spatial_dim),
            nn.Linear(spatial_dim, spatial_dim*2),
            nn.GELU(),
            nn.Linear(spatial_dim*2, spatial_dim),
        )


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
            Block(
                dim=spatial_dim, num_heads=num_heads, mlp_ratio=spatial_mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            BiTemporalMambaBlockV2(
                dim=temporal_dim, mlp_ratio=temporal_mlp_ratio, drop=drop_rate,
                d_state=d_state, d_conv=d_conv, expand=expand
            ) for _ in range(depth)
        ])

        self.Spatial_norm = norm_layer(spatial_dim)
        self.Temporal_norm = norm_layer(temporal_dim)

        self.s2t = nn.Linear(spatial_dim, temporal_dim)
        self.t2s = nn.Linear(temporal_dim, spatial_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Linear(temporal_dim , out_dim),
        )

        self.residual_ssm = ResidualSSMHead(
            coord_dim=3, d_model=residual_d_model, mlp_ratio=2.0, drop=drop_rate,
            d_state=d_state, d_conv=d_conv, expand=expand
        )

    @torch.no_grad()
    def attach_kinrope(self, parents_np, pos_ids: torch.Tensor | None = None,
                       rope_eps: float | None = None, rope_base: float | None = None,
                       joints_left=None, joints_right=None,
                       depth_weight: float = 0.5, side_weight: float = 0.1):
        num_joints = int(self.Spatial_pos_embed.shape[1])
        for blk in self.STEblocks:
            blk.attn.set_kinematic_rope(
                num_joints=num_joints,
                parents_np=parents_np,
                pos_ids=pos_ids,
                joints_left=joints_left,
                joints_right=joints_right,
                depth_weight=depth_weight,
                side_weight=side_weight,
            )
            if rope_eps is not None:
                blk.attn.rope_eps = float(rope_eps)
            if rope_base is not None:
                blk.attn.rope_base = float(rope_base)

    def STE_forward(self, x_2d, x_3d, t):
        te = self.time_mlp(t)
        if self.is_train:
            x = torch.cat((x_2d, x_3d), dim=-1)
            b, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size?
            x = rearrange(x, 'b f n c  -> (b f) n c', )
            ### now x is [batch_size, receptive frames, joint_num, 2 channels]
            x = self.Spatial_patch_to_embedding(x)
            # x = rearrange(x, 'bnew c n  -> bnew n c', )
            x += self.Spatial_pos_embed
            te = te.repeat_interleave(f, dim=0)[:, None, :]   # (B*F, 1, C)
            x = x + te
        else:
            x_2d = x_2d[:,None].repeat(1,x_3d.shape[1],1,1,1)
            x = torch.cat((x_2d, x_3d), dim=-1)
            b, h, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size?
            x = rearrange(x, 'b h f n c  -> (b h f) n c', )
            x = self.Spatial_patch_to_embedding(x)
            x += self.Spatial_pos_embed
            te = te[:, None, None, :].expand(b, h, f, -1)     # (B, H, F, C)
            te = rearrange(te, 'b h f c -> (b h f) 1 c')      # (B*H*F, 1, C)
            x = x + te

        x = self.pos_drop(x)

        blk = self.STEblocks[0]
        x = blk(x)
        # x = blk(x, vis=True)

        x = self.Spatial_norm(x)
        if self.is_train:
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        else:
            x = rearrange(x, '(b h f) n cw -> (b h n) f cw', h=h, f=f)
        return x

    def TTE_foward(self, x):
        assert len(x.shape) == 3, "shape is equal to 3"
        b, f, _  = x.shape
        x = self.s2t(x)
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        blk = self.TTEblocks[0]
        x = blk(x)
        # x = blk(x, vis=True)
        # exit()

        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(1, self.block_depth):
            tteblock = self.TTEblocks[i]

            if i % self.ste_stride == 0:
                x = self.t2s(x)  # (B, F, N, spatial_dim)
                x = rearrange(x, 'b f n cw -> (b f) n cw')
                steblock = self.STEblocks[i]

                x = steblock(x)
                x = self.Spatial_norm(x)
                x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

                x = self.s2t(x)
            else:
                x = rearrange(x, 'b f n cw -> (b n) f cw')

            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        
        return x

    def forward(self, x_2d, x_3d, t):
        if self.is_train:
            b, f, n, c = x_2d.shape
        else:
            b, h, f, n, c = x_3d.shape

        x = self.STE_forward(x_2d, x_3d, t)

        x = self.TTE_foward(x)


        if self.is_train:
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
            x = self.ST_foward(x)
            x = self.head(x)
            x = x.view(b, f, n, -1)
        else:
            # changed: keep hypotheses independent
            x = rearrange(x, '(b h n) f cw -> (b h) f n cw', h=h, n=n)
            x = self.ST_foward(x)                       # (b*h, f, n, cw)
            x = self.head(x)                            # (b*h, f, n, 3)
            x = rearrange(x, '(b h) f n c -> b h f n c', h=h)

        x = self.residual_ssm(x)

        return x
