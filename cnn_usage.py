import json
import torch
import torch.nn as nn
import argparse
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the same CNNClassifier and YelpCNNModel as in training
class CNNClassifier(nn.Module):
    def __init__(self, embedding_dim, num_filters, filter_sizes, num_classes, dropout=0.5):
        super(CNNClassifier, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        conv_outputs = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(co, dim=2)[0] for co in conv_outputs]
        cat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(cat))

class YelpCNNModel(nn.Module):
    def __init__(self, pretrained_model_name, num_filters, filter_sizes, num_classes):
        super(YelpCNNModel, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        embedding_dim = self.bert.config.hidden_size
        self.cnn = CNNClassifier(embedding_dim, num_filters, filter_sizes, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        return self.cnn(embeddings)


def predict_topk(text, model, tokenizer, mlb, top_k, max_length):
    model.eval()
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Get top-k indices
    topk_idx = probs.argsort()[-top_k:][::-1]
    # Map indices to labels
    topk_labels = [mlb.classes_[i] for i in topk_idx]
    topk_probs = probs[topk_idx]
    return list(zip(topk_labels, topk_probs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Yelp CNN Top-K Inference')
    parser.add_argument('--model_path', type=str, default='yelp_cnn_model.pth')
    parser.add_argument('--classes_path', type=str, default='mlb_classes.json')
    parser.add_argument('--pretrained_model', type=str, default='distilbert-base-uncased')
    parser.add_argument('--num_filters', type=int, default=100)
    parser.add_argument('--filter_sizes', nargs='+', type=int, default=[2,3,4])
    parser.add_argument('--text', type=str, required=True, help='Business name to classify')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top categories to return')
    parser.add_argument('--max_length', type=int, default=32, help='Max token length for encoding')
    args = parser.parse_args()

    # Load label classes
    classes = json.load(open(args.classes_path, 'r'))
    mlb = MultiLabelBinarizer()
    mlb.classes_ = classes  # classes is a list, MultiLabelBinarizer will accept indexing via list

    # Initialize model and load weights
    model = YelpCNNModel(
        args.pretrained_model,
        args.num_filters,
        args.filter_sizes,
        len(mlb.classes_)
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    # Predict top-k categories
    results = predict_topk(
        args.text,
        model,
        tokenizer,
        mlb,
        args.top_k,
        args.max_length
    )

    print(f'Input text: {args.text}')
    print(f'Top {args.top_k} categories with confidence scores:')
    for label, prob in results:
        print(f'{label}: {prob:.4f}')

# Example usage:
# python yelp_cnn_inference.py --text "Joe's Pizza and Subs" --top_k 5
