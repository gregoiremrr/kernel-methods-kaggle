import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

def tokenize_sequence(seq):
    """Map a DNA sequence to a list of integer indices (A:0, C:1, G:2, T:3)."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return [mapping[ch] for ch in seq]

def tokenize_and_pad(seqs):
    """
    Tokenize a list or pandas Series of DNA sequences.
    Assumes sequences are of equal length.
    Returns a tensor of shape (n_sequences, seq_length).
    """
    tokenized = [torch.tensor(tokenize_sequence(seq), dtype=torch.long) for seq in seqs]
    return torch.stack(tokenized)

class DNADataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # If sequences are provided as a pandas Series, extract using .iloc
        seq = self.sequences.iloc[idx] if isinstance(self.sequences, pd.Series) else self.sequences[idx]
        tokenized = torch.tensor(tokenize_sequence(seq), dtype=torch.long)
        if self.labels is not None:
            label = self.labels.iloc[idx] if isinstance(self.labels, pd.Series) else self.labels[idx]
            return tokenized, label
        else:
            return tokenized

def train_learnable_kernel(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=4, min_lr=1e-7)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_model_path = "best_model.pt"
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
        train_acc = train_correct / train_total
        
        if val_loader is not None:
            model.eval()
            all_preds = []
            all_labels = []
            val_loss_total = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss_total += loss.item() * inputs.size(0)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
            val_acc = accuracy_score(all_labels, all_preds)
            val_loss = val_loss_total / len(val_loader.dataset)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} | Train Accuracy: {train_acc:.4f} - Val Accuracy: {val_acc:.4f}")
        else:
            # No validation set: track best model using training accuracy.
            if train_acc > best_val_acc:
                best_val_acc = train_acc
                torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        
        scheduler.step(epoch_loss)
    
    if val_loader is not None:
        print(f"Best Validation Accuracy: {best_val_acc:.4f}, Model saved/loaded to {best_model_path}")
    else:
        print(f"Best Train Accuracy: {best_val_acc:.4f}, Model saved/loaded to {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    
    return model