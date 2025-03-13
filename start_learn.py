import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from utils.utils import load_dataset
from utils.utils_learn import tokenize_and_pad, train_learnable_kernel, DNADataset
from models.models import SVC
from utils.utils import train_test_split, accuracy_score

from kernels.kernels_learn import LearnableKernelModel1, LearnableKernelModel2, LearnableKernelModel3

dataset_ids = [0, 1, 2]
task = "submission" # "validation" or "submission"

num_epochs = 100
batch_size = 64
dropout = 0.4
learning_rate = 1e-2
embedding_dim = 16

# Model 1: BiGRU Representation
hidden_dim = 32
num_layers = 2

# Model 2: Convolutional Representation
num_filters = [
    64, # Dataset 0
    64, # Dataset 1
    64  # Dataset 2
]
filter_sizes = [
    [3, 5, 7, 9, 11, 13], # Dataset 0
    [3, 5, 7, 9, 11],      # Dataset 1
    [3, 5, 7, 9, 11, 13]   # Dataset 2
]

# Model 3: Transformer Representation
nhead = 4
num_encoder_layers = 2
dim_feedforward = 64
representation = 'mean'

# Process each dataset separately.
all_test_predictions = []
for dataset_id in dataset_ids:
    print(f"\nProcessing Dataset {dataset_id} ...")
    Xtrk, Xtek, Ytrk = load_dataset(dataset_id)
    
    if task == "validation":
        # Split the training set into training and validation sets.
        X_train, X_val, y_train, y_val = train_test_split(
            Xtrk['seq'], Ytrk['Bound'], test_size=0.25, stratify=Ytrk['Bound']
        )
    elif task == "submission":
        # Split the training set into training and validation sets (0.10% for validation)
        X_train, X_val, y_train, y_val = train_test_split(
            Xtrk['seq'], Ytrk['Bound'], test_size=0.001, stratify=Ytrk['Bound']
        )
    
    # Create PyTorch datasets and dataloaders.
    train_dataset = DNADataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=lambda batch: (torch.stack([item[0] for item in batch]),
                                                         torch.tensor([item[1] for item in batch], dtype=torch.long)))
    
    # Create validation set for both validation and submission tasks.
    val_dataset = DNADataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=lambda batch: (torch.stack([item[0] for item in batch]),
                                                       torch.tensor([item[1] for item in batch], dtype=torch.long)))
    
    # Initialize and train the learnable kernel model.
    # model = LearnableKernelModel1(vocab_size=4, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers,
    #                               dropout=dropout, representation='last', num_classes=2)
    model = LearnableKernelModel2(vocab_size=4, embedding_dim=embedding_dim, num_filters=num_filters[dataset_id], filter_sizes=filter_sizes[dataset_id],
                                  dropout=dropout, num_classes=2)
    # model = LearnableKernelModel3(vocab_size=4, embedding_dim=embedding_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
    #                               dim_feedforward=dim_feedforward, dropout=dropout, representation=representation, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_learnable_kernel(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device
    )
    
    # --- Compute the Precomputed Kernel Matrix from the Learned Representation ---
    # Compute features for the training set.
    F_train = model.representation_model(tokenize_and_pad(X_train).to(device)).cpu().detach().numpy()
    
    # Train an SVC using the kernel matrix computed as F_train * F_train^T.
    clf = SVC(kernel='precomputed')
    K_train = np.dot(F_train, F_train.T)
    clf.fit(K_train, y_train)
    
    # Compute validation accuracy using the small validation set.
    F_val = model.representation_model(tokenize_and_pad(X_val).to(device)).cpu().detach().numpy()
    K_val = np.dot(F_val, F_train.T)
    y_val_pred = clf.predict(K_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Final Validation Accuracy on Dataset {dataset_id}: {val_accuracy:.4f}")
    
    if task == "submission":
        # For submission, compute features on the test set.
        F_test = model.representation_model(tokenize_and_pad(Xtek['seq']).to(device)).cpu().detach().numpy()
        K_test = np.dot(F_test, F_train.T)
        y_test_pred = clf.predict(K_test)
        all_test_predictions.extend(y_test_pred.tolist())

if task == "submission":
    # Create the submission file with the required format.
    submission_df = pd.DataFrame({
        "Id": np.arange(len(all_test_predictions)),  # Id from 0 to N-1 (e.g., 0 to 2999)
        "Bound": all_test_predictions
    })
    submission_df.to_csv("Yte.csv", index=False)
    print("\nSubmission file 'submission.csv' has been saved.")
