
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import mul
from einops import rearrange


class FeedForward(nn.Module):
    """
        GDFN in Restormer: [github] https://github.com/swz30/Restormer

        INPUT = (B, D, H, W, C)
    """
    def __init__(self, dim, hidden_features):
        super(FeedForward, self).__init__()

        self.dim = dim
        self.hidden_features = hidden_features
        self.act = nn.GELU()

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
        self.project_out = nn.Conv2d(hidden_features//2, dim, kernel_size=1)

    def forward(self, x):
        B, D, H, W, C = x.shape
        x = x.permute(0,1,4,2,3).view(B*D, C, H, W)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.project_out(x)
        x = x.view(B, D, C, H, W).permute(0,1,3,4,2)
        return x


class SSTGFAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size, num_heads, qk_scale=None):
        super().__init__()
        self.dim = dim // 4
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.idx = idx


        if idx == 2:
            H_sp, W_sp, D_sp = 1, 1, 4
            self.dim = dim
        elif idx == 0:
            H_sp, W_sp, D_sp = self.resolution[0], self.split_size, 1
        elif idx == 1:
            W_sp, H_sp, D_sp = self.resolution[1], self.split_size, 1
        elif idx == 3:
            self.dim = dim
        elif idx == 4:
            self.dim = dim
        else:
            print ("ERROR MODE", idx)
            exit(0)

        self.proj = nn.Linear(self.dim, self.dim)
        self.get_v = nn.Conv3d(self.dim, self.dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=self.dim)

    def get_lepe(self, x, func, cs_windows):
        B, D, H, W, C = x.shape
        v = img2windows(x, cs_windows, self.num_heads)

        x = x.permute(0, 4, 1, 2, 3).contiguous()  # B, C, D, H, W
        lepe = func(x)  # B, C, D, H, W
        lepe = lepe.permute(0, 2, 3, 4, 1).contiguous()  # B, D, H, W, C

        lepe = img2windows(lepe, cs_windows, self.num_heads)

        return v, lepe

    def forward(self, qkv):
        """
        x: B L C
        x: Input feature, tensor size : 3, B, D, H, W, C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]
        B, D_, H_, W_, C = q.shape
        # print(q.shape)

        if self.idx == 2:
            H_sp, W_sp, D_sp = 1, 1, 4
        elif self.idx == 0:
            H_sp, W_sp, D_sp = H_, self.split_size, 1
        elif self.idx == 1:
            W_sp, H_sp, D_sp = W_, self.split_size, 1
        elif self.idx == 4:
            H_sp, W_sp, D_sp = H_, self.split_size, 1
        elif self.idx == 3:
            W_sp, H_sp, D_sp = W_, self.split_size, 1
        else:
            print ("ERROR MODE---forward", self.idx)
            exit(0)
        cs_windows = (D_sp, H_sp, W_sp)
        # print('cs_windows =', cs_windows)

        q = img2windows(q, cs_windows, self.num_heads)
        # q = q.reshape(-1, q.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k = img2windows(k, cs_windows, self.num_heads)
        # k = k.reshape(-1, k.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous() # B' head N C'

        v, lepe = self.get_lepe(v, self.get_v, cs_windows)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B' head N C' @ B' head C' N --> B' head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)

        attn_cs = (attn @ v) + lepe  # B' head N N @ B' head N C'
        x = attn_cs.transpose(1, 2).reshape(-1, q.shape[2], C)  # B' H_sp*W_sp*D_sp  head C' --> B' H_sp*W_sp*D_sp C
        # print(x.shape,'------------------')
        x = self.proj(x)

        ### Window2Img
        x = x.view(-1, *(cs_windows + (C,)))
        # print(x.shape,'-11111111111111')
        img = windows2img(x, cs_windows, B, D_, H_, W_)  # B, D, H, W, C

        return img

def img2windows(img, window_size, num_heads):
    """
    img: B, D, H, W, C
    window_size: D H W
    return: -1, [0]*[1]*[2] C --->  # B' head N C'
    """
    B, D, H, W, C = img.shape
    x = img.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    # 重新排列 permute； contiguous 紧密排列； view 同reshape
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    windows = windows.reshape(-1, windows.shape[1], num_heads, C // num_heads).permute(0, 2, 1, 3).contiguous()
    return windows

def windows2img(cs_windows, window_size, B, D, H, W):
    """
    Args:
        cs_windows: (B*num_windows, window_size, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = cs_windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0],
                        window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class PXUN(nn.Module):
    def __init__(self):
        super().__init__()
        self.PXun = nn.PixelUnshuffle(2)
    def forward(self,x):
        x = x.permute(0, 1, 2, 5, 3, 4)  # 3 B T C H W
        x = self.PXun(x)
        out = x.permute(0, 1, 2, 4, 5, 3)  # 3 B T H//2 W//2 C*4
        return out


class PX(nn.Module):
    def __init__(self):
        super().__init__()
        self.px = nn.PixelShuffle(2)
    def forward(self,x):
        x = x.permute(0, 1, 4, 2, 3)  # B T C H W
        x = self.px(x)
        out = x.permute(0, 1, 3, 4, 2)  # B T H//2 W//2 C*4
        return out


class GFBlock(nn.Module):

    def __init__(self, dim, num_heads, split_size, stage4=False, resolution=(224, 224), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.resolution = resolution
        self.stage4 = stage4
        self.norm1 = norm_layer(dim)

        self.attns = nn.ModuleList([
            SSTGFAttention(
                dim, resolution, idx=i, split_size=split_size, num_heads=num_heads, qk_scale=qk_scale)
            for i in range(5)])

        self.norm2 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim=dim, hidden_features=mlp_hidden_dim)
        self.PXun = PXUN()
        self.PX = PX()

    def forward_part1(self, x):
        B, D, H, W, C = x.shape

        window_size_p = (1, 2*self.split_size, 2*self.split_size)
        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size_p[0] - D % window_size_p[0]) % window_size_p[0]
        pad_b = (window_size_p[1] - H % window_size_p[1]) % window_size_p[1]
        pad_r = (window_size_p[2] - W % window_size_p[2]) % window_size_p[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, Dp, Hp, Wp, C, 3).permute(5, 0, 1, 2, 3, 4)  # 3, _, Dp, Hp, Wp, _

        x1 = self.attns[0](qkv[:, :, :, :, :, :C//4])
        x2 = self.attns[1](qkv[:, :, :, :, :, C//2:C*3//4])  # B T H W C

        x1_px = self.PXun(qkv[:, :, :, :, :, C//4:C//2] + x1)
        x2_px = self.PXun(qkv[:, :, :, :, :, C*3//4:] + x2)


        x1_out = self.attns[3](x1_px)
        x2_out = self.attns[4](x2_px)


        x11 = self.PX(x1_out)
        x22 = self.PX(x2_out)

        attened_x = torch.cat([x1, x11, x2, x22], dim=-1)

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            attened_x = attened_x[:, :D, :H, :W, :].contiguous()
        ######################point attn###########################
        qkv = self.qkv(attened_x).reshape(B, D, H, W, C, 3).permute(5, 0, 1, 2, 3, 4)  # 3, _, D, H, W, _
        # print(qkv.shape)
        x3 = self.attns[2](qkv)

        return x3

    def forward_part2(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        shortcut = x
        x = self.forward_part1(x)
        x = shortcut + x
        x = x + self.forward_part2(x)
        return x

class BasicLayer(nn.Module):

    def __init__(self,
                 dim,
                 stage4,
                 depth,
                 num_heads,
                 split_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            GFBlock(
                dim=dim,
                num_heads=num_heads,
                split_size=split_size,
                stage4=stage4,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
            )
            for i in range(depth)])

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b d h w c')

        for i, blk in enumerate(self.blocks):
            x = blk(x)
        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b c d h w')
        # print(x.shape)
        return x

class Layer(nn.Module):
    def __init__(self, plane, depth, num_heads, split_size, stage4):
        super(Layer, self).__init__()
        self.upper = BasicLayer(plane, stage4, depth=depth, num_heads=num_heads, split_size=split_size)

    def forward(self, x):
        out = self.upper(x)
        return out
