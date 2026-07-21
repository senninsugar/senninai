import torch

def generate(model, input_ids, max_new_tokens, temperature=0.7, top_k=40):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = input_ids[:, -model.config["max_position_embeddings"]:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, idx_next), dim=1)
        if idx_next.item() == model.config.get("eos_token_id", -1):
            break
    return input_ids
