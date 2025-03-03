import torch
from torch.utils.data import DataLoader, Dataset
class Trainer:
    def __init__(self, model: torch.nn.Module, train_data: DataLoader, optimizer: torch.optim.Optimizer, gpu_id: int, save_every: int):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        
    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        out = self.model(source)
        loss = torch.nn.CrossEntropyLoss()(out, targets)
        loss.backward()
        self.optimizer.step()      
    
    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
    
    def _save_checkpoint(self, epoch):
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, "checkpoint.pt")
        print(f"Epoch {epoch} | chckpoin saved at checkpoint.pt")
        
    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
    
# def  