import torch
import os

class Trainer:
    def __init__(self, model, dataloader, optimizer, device, config):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=config.get("pad_token_id", 0))

    def train(self):
        self.model.train()
        for epoch in range(self.config["epochs"]):
            total_loss = 0
            for step, (x, y) in enumerate(self.dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if step % self.config["log_interval"] == 0:
                    print(f"Epoch {epoch+1}/{self.config['epochs']} | Step {step} | Loss: {loss.item():.4f}")

            os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(self.config["checkpoint_dir"], f"model_epoch_{epoch+1}.pt"))
