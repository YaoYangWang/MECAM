import torch
from torch import nn
from einops import rearrange
from torch import einsum

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        text_dim,  # 文本的维度，例如 768
        audio_dim,  # 音频的维度，例如 16
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        talking_heads = False,
        prenorm = False,
    ):
        super().__init__()

        self.norm_text = nn.LayerNorm(text_dim) if prenorm else nn.Identity()
        self.norm_audio = nn.LayerNorm(audio_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        audio_inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)

        # 针对音频的线性变换（查询和键）
        self.audio_to_qk = nn.Linear(audio_dim, audio_inner_dim, bias = False)

        # 针对文本的线性变换（值）
        self.text_to_v = nn.Linear(text_dim, audio_inner_dim, bias = False)

        # 输出变换以匹配音频维度
        self.to_out = nn.Linear(audio_inner_dim, audio_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()

    def forward(
        self,
        text,
        audio,
        mask = None,
        text_mask = None,
        return_attn = False
    ):
        b, h, device = audio.shape[0], self.heads, audio.device

        text = self.norm_text(text)
        audio = self.norm_audio(audio)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # 转换为 [batch_size, 1, audio_feature_dim]
        if text.dim() == 2:
            text = text.unsqueeze(1)  # 转换为 [batch_size, 1, audio_feature_dim]


        # 获取音频的查询和键，文本的值
        qk = self.audio_to_qk(audio)
        v = self.text_to_v(text)

        # 分离头部
        qk, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (qk, v))

        # 计算相似度
        sim = einsum('b h i d, b h j d -> b h i j', qk, v) * self.scale

        # 注意力计算和输出
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)
        attn = self.talking_heads(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        if audio.dim() == 3 and audio.shape[1] == 1:
            out = out.squeeze(1)  # 转换回 [batch_size, audio_feature_dim
        if return_attn:
            return out, attn

        return out
