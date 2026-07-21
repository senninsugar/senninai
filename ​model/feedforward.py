import torch

class SwiGLU(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        self.w1 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = torch.nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
