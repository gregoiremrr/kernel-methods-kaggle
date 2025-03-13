import torch
import torch.nn as nn
import torch.nn.functional as F

class BiGRURepresentation(nn.Module):
    def __init__(self, vocab_size=4, embedding_dim=16, hidden_dim=64, num_layers=1, dropout=0.1, representation='last'):
        """
        A Bi-directional GRU to extract sequence representations.
        
        Args:
            representation: 'last' returns the final hidden states (concatenated),
                            'mean' returns the average of all outputs.
        """
        super(BiGRURepresentation, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bigru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        self.representation = representation

    def forward(self, x):
        emb = self.embedding(x)  # (batch, seq_len, embedding_dim)
        gru_out, hidden = self.bigru(emb)  # gru_out: (batch, seq_len, hidden_dim*2)
        if self.representation == 'last':
            # Concatenate the last forward and last backward hidden states.
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            return last_hidden  # (batch, hidden_dim*2)
        elif self.representation == 'mean':
            mean_hidden = torch.mean(gru_out, dim=1)
            return mean_hidden  # (batch, hidden_dim*2)
        else:
            return gru_out

class LearnableKernelModel1(nn.Module):
    def __init__(self, vocab_size=4, embedding_dim=16, hidden_dim=64, num_layers=1, dropout=0.1, representation='last', num_classes=2):
        """
        This model learns a representation (the "kernel") and a classifier.
        After training, the representation is used to compute a kernel matrix.
        """
        super(LearnableKernelModel1, self).__init__()
        self.representation_model = BiGRURepresentation(vocab_size, embedding_dim, hidden_dim, num_layers, dropout, representation)
        self.layer_norm = nn.LayerNorm(hidden_dim*2)
        self.classifier = nn.Linear(hidden_dim*2, num_classes)
    
    def forward(self, x):
        rep = self.representation_model(x)
        rep = self.layer_norm(rep)
        logits = self.classifier(rep)
        return logits


class LearnableKernelModel2(nn.Module):
    def __init__(self, vocab_size=4, embedding_dim=16, num_filters=64, filter_sizes=[3,5,7], dropout=0.1, num_classes=2):
        """
        A CNN-based model for DNA sequences.
        
        Args:
            vocab_size (int): Number of tokens (for DNA, typically 4: A, C, G, T).
            embedding_dim (int): Dimension of the embedding space.
            num_filters (int): Number of filters per convolution.
            filter_sizes (list): List of kernel sizes (e.g. [3,5,7]).
            dropout (float): Dropout rate.
            num_classes (int): Number of output classes.
        """
        super(LearnableKernelModel2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        # One BatchNorm1d per conv layer.
        self.bn_convs = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        emb = emb.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        conv_outputs = []
        for conv, bn in zip(self.convs, self.bn_convs):
            c_out = conv(emb)  # (batch_size, num_filters, L_out)
            c_out = bn(c_out)
            c_out = F.relu(c_out)
            # Global max pooling over the temporal dimension.
            pooled = torch.max(c_out, dim=2)[0]  # (batch_size, num_filters)
            conv_outputs.append(pooled)
        features = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        features = self.dropout(features)
        logits = self.fc(features)
        return logits

    def representation_model(self, x):
        """
        Returns the CNN-based representation.
        """
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        emb = emb.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        conv_outputs = []
        for conv, bn in zip(self.convs, self.bn_convs):
            c_out = conv(emb)  # (batch_size, num_filters, L_out)
            c_out = bn(c_out)
            c_out = F.relu(c_out)
            pooled = torch.max(c_out, dim=2)[0]
            conv_outputs.append(pooled)
        features = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        return features


class LearnableKernelModel3(nn.Module):
    def __init__(self, vocab_size=4, embedding_dim=16, nhead=4, num_encoder_layers=2,
                 dim_feedforward=64, dropout=0.1, representation='mean', num_classes=2):
        """
        A Transformer-based model for DNA sequences.
        
        Args:
            vocab_size (int): Number of tokens.
            embedding_dim (int): Dimension of the embedding space.
            nhead (int): Number of heads in the multi-head attention.
            num_encoder_layers (int): Number of Transformer encoder layers.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
            representation (str): 'mean' for average pooling, 'max' for max pooling,
                                  or any other option to select the first token.
            num_classes (int): Number of output classes.
        """
        super(LearnableKernelModel3, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.representation = representation
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        # Transformer expects shape: (seq_len, batch_size, embedding_dim)
        emb = emb.transpose(0, 1)
        transformer_out = self.transformer_encoder(emb)  # (seq_len, batch_size, embedding_dim)
        transformer_out = transformer_out.transpose(0, 1)  # (batch_size, seq_len, embedding_dim)
        
        if self.representation == 'mean':
            rep = transformer_out.mean(dim=1)
        elif self.representation == 'max':
            rep, _ = transformer_out.max(dim=1)
        else:
            # Default: use the representation of the first token.
            rep = transformer_out[:, 0, :]
        rep = self.layer_norm(rep)
        logits = self.fc(rep)
        return logits
    
    def representation_model(self, x):
        """
        Returns the Transformer-based representation.
        """
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        emb = emb.transpose(0, 1)  # (seq_len, batch_size, embedding_dim)
        transformer_out = self.transformer_encoder(emb)  # (seq_len, batch_size, embedding_dim)
        transformer_out = transformer_out.transpose(0, 1)  # (batch_size, seq_len, embedding_dim)
        
        if self.representation == 'mean':
            rep = transformer_out.mean(dim=1)
        elif self.representation == 'max':
            rep, _ = transformer_out.max(dim=1)
        else:
            rep = transformer_out[:, 0, :]
        rep = self.layer_norm(rep)
        return rep
