import json
import torch
from utils.device import get_cpu_device
from tokenizer.tokenizer import SimpleTokenizer
from model.transformer import TinyLLM
from model.generate import generate

def run_chat(checkpoint_path):
    device = get_cpu_device()

    with open("config/model.json") as f:
        model_config = json.load(f)

    tokenizer = SimpleTokenizer("config/tokenizer.json")
    model = TinyLLM(model_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    while True:
        prompt = input("User > ")
        if prompt.lower() in ["exit", "quit"]:
            break

        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        output_ids = generate(model, input_ids, max_new_tokens=100)
        response = tokenizer.decode(output_ids[0].tolist())
        print(f"Bot > {response}")

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/model_epoch_10.pt"
    run_chat(ckpt)
