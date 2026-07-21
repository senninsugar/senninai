import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        self.samples = []
        for line in lines:
            ids = tokenizer.encode(line)
            if len(ids) > 1:
                self.samples.append(ids[:max_seq_len])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, pad_token_id):
    max_len = max(len(item) for item in batch)
    input_ids = []
    labels = []
    
    for item in batch:
        padded = item + [pad_token_id] * (max_len - item)
        input_ids.append(padded[:-1])
        labels.append(padded[1:])
        
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def get_dataloader(file_path, tokenizer, max_seq_len, batch_size, pad_token_id):
    dataset = TextDataset(file_path, tokenizer, max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_token_id))
