import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_
from functools import lru_cache
from einops import rearrange
import time


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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

class ChannelAttention3D(nn.Module):
    """Channel attention for video processing."""
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention3D, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # Pool across spatial and temporal dimensions
            nn.Conv3d(num_feat, num_feat // squeeze_factor, (1, 1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat // squeeze_factor, num_feat, (1, 1, 1), padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.attention(x)
        return x * y
    
class CAB3D(nn.Module):
    """Channel Attention Block adapted for video processing."""
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB3D, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv3d(num_feat, num_feat // compress_ratio, (1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(num_feat // compress_ratio, num_feat, (1, 3, 3), stride=1, padding=(0, 1, 1)),
            ChannelAttention3D(num_feat, squeeze_factor)
        )
    def forward(self, x):
        return self.cab(x)


def window_partition(x, window_size):
    """
    Args:
        x: (b, d, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size[0] * window_size[1] * window_size[2], c)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2],
                                                                  C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, wD* wH* wW, C)
        window_size (tuple[int]): Window size (wD, wH, wW)
        B (int): Batch size
        D (int): number of frames
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads  
        head_dim = dim // num_heads  
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02) #weight initializer for truncated normal distribution
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wd*Wh*Ww (N), Wd*Wh*Ww (N)) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, num_heads, N, C
        #print("this x",x.shape)
        q = q * self.scale
        #print("this is q:",q.shape)
        attn = (q @ k.transpose(-2, -1))  # B_, nH, N, N
        #print("this is atten map:",attn.shape)
        relative_position_bias = self.relative_position_bias_table[rpi[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # N,N,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, N, N
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N
        #print("this is atten:",attn.shape)
        if mask is not None:
            nW = mask.shape[0]
            #print("this is mask:",mask.shape)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0) #(1, nw, 1, n, n)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x #B_,N,C

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, n):
        #need to be re-calculated 
        # calculate flops for 1 window with token length of n
        flops = 0
        flops += n * self.dim * 3 * self.dim
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        flops += n * self.dim * self.dim
        return flops


class HAB(nn.Module):
    r""" Hybrid Attention Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=(2, 7, 7),
                 shift_size=(0, 0, 0),
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 num_frames=5):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.num_frames = num_frames
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        # assert 0 <= self.shift_size[0] < self.window_size[0], 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        
        self.conv_scale = conv_scale
        self.conv_block = CAB3D(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, t, h, w, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"
        #print("this is x input of swin",x.shape)
        shortcut = x

        x = self.norm1(x)

        # Conv_X
        conv_x = self.conv_block(x.permute(0, 4, 1, 2, 3)) # b c t h w
        conv_x = conv_x.permute(0, 2, 3, 4, 1).contiguous() # b t h w c

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.window_size[0] - t % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - h % self.window_size[1]) % self.window_size[1]
        pad_r = (self.window_size[2] - w % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        #print("this is x after pad of swin",x.shape)
        
        # cyclic shift
        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x,
                                     self.window_size)  # nw*b, window_size[0]*window_size[1]*window_size[2], c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        #print("this is shift_size", self.shift_size)
        #print("this is x_window in Swin",x_windows.shape)
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)  # B*nW, N, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(self.window_size + [
            c,
        ]))
        shifted_x = window_reverse(attn_windows, self.window_size, b, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in self.shift_size):
            x = torch.roll(
                shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :t, :h, :w, :].contiguous()

        # FFN
        x = shortcut + self.drop_path(x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # b,t,h,w,c
        return x

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, '
                f'window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}')

    def flops(self):
        flops = 0
        h, w = self.input_resolution
        # norm1
        flops += self.dim * h * w * self.num_frames
        # W-MSA/SW-MSA
        nw = h * w / self.window_size[1] / self.window_size[2]
        flops += nw * self.attn.flops(self.window_size[1] * self.window_size[2] * self.num_frames)
        # mlp
        flops += 2 * self.num_frames * h * w * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * h * w * self.num_frames
        return flops

class Overlapping_window_partition(nn.Module):
    """
        Arg:
            window_size tuple(int): Window size.
            overlap_win_size tuple(int): Overlapping Window size.
    """
    def __init__(self, window_size, overlap_win_size):
        super().__init__()
        self.window_size=window_size
        self.overlap_win_size = overlap_win_size
        self.padding= (0,(overlap_win_size[1]-window_size[1])//2,(overlap_win_size[2]-window_size[2])//2)

    
    def forward(self, x): 

        B, C, D, H, W = x.shape
        # Input shape: (B, C, D, H, W)
        x = F.pad(x,
                    (self.padding[2], self.padding[2],
                    self.padding[1], self.padding[1],
                    self.padding[0], self.padding[0])
                    )
        # print("after pad:", x.shape)

        x = (x
                .unfold(2, size=self.overlap_win_size[0], step=self.window_size[0])
                .unfold(3, size=self.overlap_win_size[1], step=self.window_size[1])
                .unfold(4, size=self.overlap_win_size[2], step=self.window_size[2])
                .permute(0, 2, 3, 4, 1, 5, 6, 7)
                .reshape(B, -1, C * np.prod(self.overlap_win_size))
                .transpose(1, 2)
                )  
        #B, C*owd*owh*oww, nW( D//wd * H//wh * W//ww)       
        assert x.shape[-1] == D//self.window_size[0] * H//self.window_size[1] * W//self.window_size[2], print("Shape[-1] is wrong, window size H,W must be 4*n where n=1,2,3,4,...")
        assert x.shape[-2] == C*self.overlap_win_size[0]*self.overlap_win_size[1]*self.overlap_win_size[2], print("#Shape[-2] is wrong, window size H,W must be 4*n where n=1,2,3,4,...")
        return x
    
    

class OCAB(nn.Module):
    # overlapping cross-attention block

    def __init__(self, 
                dim,
                input_resolution,
                window_size,
                overlap_ratio,
                num_heads,
                qkv_bias=True,
                qk_scale=None,
                mlp_ratio=2,
                norm_layer=nn.LayerNorm,
                attn_drop=0., 
                drop_path = 0.,
                proj_drop=0.
                ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = (1, window_size[1], window_size[2]) #set frame size to 1, focusing on spatial information
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        
        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3,  bias=qkv_bias)

        overlap_D_size = 1  #no overlap in temporal dimension, focusing on spatial information
        overlap_H_size = int(window_size[1] * overlap_ratio) + window_size[1]
        overlap_W_size = int(window_size[2] * overlap_ratio) + window_size[2]
        self.overlap_win_size = (overlap_D_size, overlap_H_size, overlap_W_size)
        self.kv_partition = Overlapping_window_partition(window_size=self.window_size, overlap_win_size= self.overlap_win_size)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size[1] + self.overlap_win_size[1] - 1) * (window_size[2] + self.overlap_win_size[2] - 1), num_heads))  #(owh+wd-1) (owd*wh-1), nH
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x, x_size, rpi_oca):
        h, w = x_size
        b, t, h, w, c = x.shape

        shortcut = x
        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_b = (self.window_size[1] - h % self.window_size[1]) % self.window_size[1]
        pad_r = (self.window_size[2] - w % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, 0, 0))
        _, _, Hp, Wp, _ = x.shape

        qkv = self.qkv(x).reshape(b, t, Hp, Wp, 3, c).permute(4, 0, 5, 1, 2, 3) # 3, b, c, t, Hp, Wp
        
        #partition q windows
        q = qkv[0].permute(0, 2, 3, 4, 1) # b, t, Hp, Wp, c
        q_windows = window_partition(q, self.window_size)  # nw*b, wd*wh*ww, c
        #partition kv windows
        kv = torch.cat((qkv[1], qkv[2]), dim=1) # b, 2*c, t, Hp, Wp
        kv_windows = self.kv_partition(kv) # b, 2*c*owd*owh*oww, nw
        kv_windows = rearrange(kv_windows, 'b (nc c owd owh oww) nw -> nc (b nw) (owd owh oww) c', nc=2, c=c, owd=self.overlap_win_size[0], owh=self.overlap_win_size[1], oww=self.overlap_win_size[2]).contiguous() # 2, nw*b, owd*owh*oww, c
        k_windows, v_windows = kv_windows[0], kv_windows[1] #each with nw*b, owd*owh*oww, c

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim// self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, nq, d
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) #nw*b, num_heads, nq, n #(nq = prod(win_size), n = prod(overlap_win_size))

        relative_position_bias = self.relative_position_bias_table[rpi_oca.view(-1)].view(
            np.prod(self.window_size), np.prod(self.overlap_win_size), -1)  # nq, n, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, nq, n
        attn = attn + relative_position_bias.unsqueeze(0)

        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)
        x = window_reverse(attn_windows, self.window_size, b, t, Hp, Wp)  # b Dp Hp Wp c
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :t, :h, :w, :].contiguous()

        x = self.drop_path(x) + shortcut

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #b, t, h, w, c
        return x


class AttentionBlocks(nn.Module):
    """ A series of attention blocks for one RHAG.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 num_frames=5):

        super().__init__()
        self.dim = dim  
        self.input_resolution = input_resolution  #64,64
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            HAB(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0], window_size[1] // 2, window_size[2] // 2), #alternating shift
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                num_frames=num_frames) for i in range(depth)
        ])

        # OCAB
        self.overlap_attn = OCAB(
                            dim=dim,
                            input_resolution=input_resolution,
                            window_size=window_size,
                            overlap_ratio=overlap_ratio,
                            num_heads=num_heads,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            mlp_ratio=mlp_ratio,
                            norm_layer=norm_layer
                            )
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, params):
        #x (b,c,t,h,w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  #(b,t,h,w,c)
        #SWA
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x,x_size, params['rpi_sa'], params['attn_mask'])
            else:
                #print("this is x in blk",x.shape)
                x = blk(x, x_size, rpi_sa = params['rpi_sa'], attn_mask = params['attn_mask'])
        
        #OCA
        x = self.overlap_attn(x, x_size, params['rpi_oca'])
        
        if self.downsample is not None:
            x = self.downsample(x)

        x = x.permute(0, 4, 1, 2, 3).contiguous()  #b,c,t,h,w
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
            #print("swinlayer",blk.flops()/1e9)
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RHAG(nn.Module):
    """Multi-frame Residual Hybrid Attention Group (RHAG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=(1, 1),
                 resi_connection='1conv',
                 num_frames=5):
        super(RHAG, self).__init__()

        self.dim = dim  
        self.input_resolution = input_resolution  
        self.num_frames=num_frames
        self.residual_group = AttentionBlocks(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            num_frames=num_frames)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size,attn_mask):
        n, c, t, h, w = x.shape
        x_ori = x
        #print("this is x in RSTB", x.shape)
        x = self.residual_group(x, x_size,attn_mask)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        x = self.conv(x)
        x = x.view(n, t, -1, h, w)
        x = self.patch_embed(x)
        x = x + x_ori
        #n, c, t, h, w
        return x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.num_frames * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        #flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size=1, in_chans=3, embed_dim=96, num_frames=5, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.num_frames = num_frames

        self.in_chans = in_chans  
        self.embed_dim = embed_dim  

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        #x.size = (n,t,embed_dim,h,w)
        n, t, c, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        if self.norm is not None:
            x = x.flatten(2).transpose(1, 2) 
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, t, h, w)
        #n c t h w
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim *self.num_frames
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size=1, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


@lru_cache()
def compute_mask(t, x_size, window_size, shift_size, device):

    h, w = x_size
    #print([t, x_size, window_size, shift_size])
    Dp = int(np.ceil(t / window_size[0])) * window_size[0]
    Hp = int(np.ceil(h / window_size[1])) * window_size[1]
    Wp = int(np.ceil(w / window_size[2])) * window_size[2]
    img_mask = torch.zeros((1, Dp, Hp, Wp, 1), device=device)  # 1 h w 1
    #print(img_mask.shape)
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    # print(mask_windows.shape)
    #mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    mask_windows = mask_windows.view(-1, window_size[0] * window_size[1] * window_size[2])
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    # print(attn_mask.shape)
    return attn_mask

class MFHAT(nn.Module):
    r""" Multi-Frame Hybrid Attention Transformer 
    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: (2, 7, 7)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
        num_frames: The number of frames processed in the propagation block in PSRT-recurrent
    """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=(2, 7, 7),
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 overlap_ratio=0.5,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=4,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 num_frames=3,
                 **kwargs):
        super(MFHAT, self).__init__()
        num_in_ch = in_chans  #3
        num_out_ch = in_chans  #3
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        self.window_size = window_size
        self.shift_size = (window_size[0], window_size[1] // 2, window_size[2] // 2)
        self.overlap_ratio = overlap_ratio #change?
        self.num_frames = num_frames

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        relative_position_index_OCA = self.calculate_rpi_oca_2D()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_first_feat = nn.Conv2d(num_feat, embed_dim, 3, 1, 1)
        
        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)  
        self.embed_dim = embed_dim  
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim  
        self.mlp_ratio = mlp_ratio  

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            num_frames=num_frames,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches  #64*64
        patches_resolution = self.patch_embed.patches_resolution  #[64,64]
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build RHAG blocks
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHAG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio = overlap_ratio,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                num_frames=num_frames)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            # self.conv_before_upsample = nn.Sequential(
            #     nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.conv_before_upsample = nn.Conv2d(embed_dim, num_feat, 3, 1, 1)
            #self.conv_before_recurrent_upsample = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            #self.upsample = Upsample(upscale, num_feat)
            #self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self): #change
        # calculate relative position index for SA
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        return relative_position_index

    def calculate_rpi_oca_2D(self):
    # calculate relative position index for OCA
        window_size_H_ext = self.window_size[1] + int(self.overlap_ratio * self.window_size[1])
        window_size_W_ext = self.window_size[2] + int(self.overlap_ratio * self.window_size[2])

        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_H_ext)
        coords_w = torch.arange(window_size_W_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]   # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += self.window_size[1] - window_size_H_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[2] - window_size_W_ext + 1

        relative_coords[:, :, 0] *= self.window_size[2] + window_size_W_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):

        x_size = (x.shape[3], x.shape[4])  #180,320
        h, w = x_size
        #print("x_size:",x_size)
        x = self.patch_embed(x)  #n,embed_dim,t,h,w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        attn_mask = compute_mask(self.num_frames,x_size,tuple(self.window_size),self.shift_size,x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA, 'rpi_oca': self.relative_position_index_OCA}
        # print("this is attn_mask", attn_mask.shape)
        for layer in self.layers:
            # print("this is x before layer", x.shape)
            x = layer(x.contiguous(), x_size , params)

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)  # b seq_len c

        x = x.permute(0, 1, 4, 2, 3).contiguous()

        return x

    def forward(self, x, ref=None):
        n, t, c, h, w = x.size()

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            if c == 3:
                x = x.view(-1, c, h, w)
                x = self.conv_first(x)
                #x = self.feature_extraction(x)
                x = x.view(n, t, -1, h, w)

            if c == 64:
                x = x.view(-1, c, h, w)
                x = self.conv_first_feat(x)
                x = x.view(n, t, -1, h, w)

            x_center = x[:, t // 2, :, :, :].contiguous()
            feats = self.forward_features(x)

            x = self.conv_after_body(feats[:, t // 2, :, :, :]) + x_center
            if ref:
                x = self.conv_before_upsample(x)
            #x = self.conv_last(self.upsample(x))

        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        #flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i,layer in enumerate(self.layers):
            layer_flop=layer.flops()
            flops += layer_flop
            print(i,layer_flop / 1e9)


        flops += h * w * self.num_frames * self.embed_dim
        flops += h * w * 9 * self.embed_dim * self.embed_dim

        #flops += self.upsample.flops()
        return flops




class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.dim
        flops += (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim
        return flops


if __name__ == '__main__':
    upscale = 4
    window_size = (2, 8, 8)
    height = (256 // upscale // window_size[1] + 1) * window_size[1]
    width = (256 // upscale // window_size[2] + 1) * window_size[2]

    model = MFHAT(
        img_size=height,
        patch_size=1,
        in_chans=3,
        embed_dim=96,
        depths=[1,2,3],
        num_heads=[4,4,4],
        window_size=window_size,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=4,
        img_range=1.,
        upsampler='pixelshuffle',
        resi_connection='1conv',
        num_frames=3
    )

    # print(model.summary)
    #print(height, width, model.flops() / 1e9)
    print(height)
    x = torch.randn((2, 3, 3, height, width))
    start_time = time.time()
    output = model(x)
    end_time = time.time()
    print(output.shape)
    print("Inference Time:", end_time - start_time)


