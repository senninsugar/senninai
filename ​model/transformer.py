import torch
from model.embedding import TokenEmbedding
from model.layernorm import RMSNorm
from model.rope import RotaryEmbedding
from model.block import TransformerBlock

class TinyLLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embeddings = TokenEmbedding(config["vocab_size"], config["hidden_size"])
        self.layers = torch.nn.ModuleList([TransformerBlock(config) for _ in range(config["num_hidden_layers"])])
        self.norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.output = torch.nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        self.rope = RotaryEmbedding(config["hidden_size"] // config["num_attention_heads"], config["max_position_embeddings"], config["rope_theta"])

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        h = self.tok_embeddings(input_ids)
        cos, sin = self.rope(h, seq_len)

        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, cos, sin, mask)

        h = self.norm(h)
        return self.output(h)
