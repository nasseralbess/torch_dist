import torch
from torch.utils.data import DataLoader, Dataset

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]
    
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
        print(f"Epoch {epoch} | checkpoint saved at checkpoint.pt")
        
    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
    
def  load_train_objs():
    train_set = MyTrainDataset(2048)
    model = torch.nn.Linear(20,1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=True)

def main(device, total_epochs, save_every):
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size=32)
    trainer = Trainer(model, train_data, optimizer, device, save_every)
    trainer.train(total_epochs)

if __name__=="__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    device = 0
    main(device, total_epochs, save_every)