import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.5):
        super(BiLSTM_CRF, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        self.num_layers = num_layers
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(tag_size, batch_first=True)
        
    def forward(self, x, mask=None, labels=None):
        x = self.word_embeddings(x)
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions
