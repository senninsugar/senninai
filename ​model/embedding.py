import torch

class TokenEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, hidden_size)

    def forward(self, x):
        return self.emb(x)
