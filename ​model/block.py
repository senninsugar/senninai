import torch
from model.layernorm import RMSNorm
from model.attention import GroupedQueryAttention
from model.feedforward import SwiGLU

class TransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.feed_forward = SwiGLU(config)

    def forward(self, x, cos, sin, mask=None):
        h = x + self.attention(self.attention_norm(x), cos, sin, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
