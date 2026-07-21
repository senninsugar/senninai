import json
from utils.seed import set_seed
from utils.device import get_cpu_device
from tokenizer.tokenizer import SimpleTokenizer
from dataset.loader import get_dataloader
from model.transformer import TinyLLM
from trainer.trainer import Trainer
import torch

def main():
    set_seed(42)
    device = get_cpu_device()

    with open("config/model.json") as f:
        model_config = json.load(f)
    with open("config/train.json") as f:
        train_config = json.load(f)

    tokenizer = SimpleTokenizer("config/tokenizer.json")
    dataloader = get_dataloader("data/train.txt", tokenizer, train_config["max_seq_len"], train_config["batch_size"], model_config["pad_token_id"])

    model = TinyLLM(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["learning_rate"], weight_decay=train_config["weight_decay"])

    trainer = Trainer(model, dataloader, optimizer, device, train_config)
    trainer.train()

if __name__ == "__main__":
    main()
