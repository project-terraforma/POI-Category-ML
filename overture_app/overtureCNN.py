import json
import torch
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OvertureDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, mlb):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = [d['name'] for d in data]
        self.labels = [d['categories'] for d in data]
        self.mlb = mlb
        self.encoded_labels = mlb.transform(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        label = torch.FloatTensor(self.encoded_labels[idx])
        return input_ids, attention_mask, label

