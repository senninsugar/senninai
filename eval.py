import json
import torch
from utils.device import get_cpu_device
from tokenizer.tokenizer import SimpleTokenizer
from dataset.loader import get_dataloader
from model.transformer import TinyLLM

def evaluate(checkpoint_path):
    device = get_cpu_device()

    with open("config/model.json") as f:
        model_config = json.load(f)
    with open("config/train.json") as f:
        train_config = json.load(f)

    tokenizer = SimpleTokenizer("config/tokenizer.json")
    dataloader = get_dataloader("data/valid.txt", tokenizer, train_config["max_seq_len"], train_config["batch_size"], model_config["pad_token_id"])

    model = TinyLLM(model_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=model_config["pad_token_id"])
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            total_steps += 1

    avg_loss = total_loss / max(total_steps, 1)
    print(f"Validation Loss: {avg_loss:.4f} | Perplexity: {torch.exp(torch.tensor(avg_loss)):.4f}")

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/model_epoch_10.pt"
    evaluate(ckpt)
