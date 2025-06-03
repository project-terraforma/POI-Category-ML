import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class YelpDataset(Dataset):
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

class CNNClassifier(nn.Module):
    def __init__(self, embedding_dim, num_filters, filter_sizes, num_classes, dropout=0.5):
        super(CNNClassifier, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):  # x: (batch_size, seq_len, embedding_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, embedding_dim)
        conv_outputs = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(co, dim=2)[0] for co in conv_outputs]
        cat = torch.cat(pooled, dim=1)
        drop = self.dropout(cat)
        return self.fc(drop)

class YelpCNNModel(nn.Module):
    def __init__(self, pretrained_model_name, num_filters, filter_sizes, num_classes):
        super(YelpCNNModel, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        embedding_dim = self.bert.config.hidden_size
        self.cnn = CNNClassifier(embedding_dim, num_filters, filter_sizes, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # (batch_size, seq_len, embed_dim)
        logits = self.cnn(embeddings)
        return logits


def load_data(json_path, sample_size=None):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    # Filter entries with categories and a name
    data = [d for d in data if d.get('categories') and d.get('name')]
    # Format categories as list
    for d in data:
        d['categories'] = [c.strip() for c in d['categories'].split(',')]
    if sample_size:
        data = data[:sample_size]
    return data


def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    # Settings
    pretrained_model = 'distilbert-base-uncased'
    json_path = './data/yelp_academic_dataset_business.json'
    sample_size = 10000  # adjust as needed
    max_length = 32
    batch_size = 32
    epochs = 3
    num_filters = 100
    filter_sizes = [2, 3, 4]
    learning_rate = 2e-5

    data = load_data(json_path, sample_size)
    all_categories = [d['categories'] for d in data]
    mlb = MultiLabelBinarizer()
    mlb.fit(all_categories)
    num_classes = len(mlb.classes_)
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = YelpDataset(train_data, tokenizer, max_length, mlb)
    test_dataset = YelpDataset(test_data, tokenizer, max_length, mlb)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = YelpCNNModel(pretrained_model, num_filters, filter_sizes, num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs+1):
        train_loss = train(model, train_loader, criterion, optimizer)
        test_loss = evaluate(model, test_loader, criterion)
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}')

    # Save model and mlb
    torch.save(model.state_dict(), 'yelp_cnn_model.pth')
    with open('mlb_classes.json', 'w') as f:
        json.dump(mlb.classes_.tolist(), f)

if __name__ == '__main__':
    main()
