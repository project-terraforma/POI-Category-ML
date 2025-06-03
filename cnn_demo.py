import json
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import MultiLabelBinarizer
import streamlit as st

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN classifier (dynamic structure)
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

# Full model
class YelpCNNModel(nn.Module):
    def __init__(self, num_filters, filter_sizes, num_classes):
        super(YelpCNNModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        embedding_dim = self.bert.config.hidden_size
        self.cnn = CNNClassifier(embedding_dim, num_filters, filter_sizes, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        return self.cnn(embeddings)

@st.cache_resource
def load_resources(model_path: str, classes_path: str):
    # Load classes and infer model params
    classes = json.load(open(classes_path, 'r'))
    mlb = MultiLabelBinarizer()
    mlb.classes_ = classes

    checkpoint = torch.load(model_path, map_location=DEVICE)
    conv_keys = sorted(
        [k for k in checkpoint.keys() if k.startswith('cnn.convs') and k.endswith('.weight')],
        key=lambda x: int(x.split('.')[2])
    )
    filter_sizes = [checkpoint[k].shape[2] for k in conv_keys]
    num_filters = checkpoint[conv_keys[0]].shape[0]
    num_classes = checkpoint['cnn.fc.bias'].shape[0]

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Build model and load weights
    model = YelpCNNModel(num_filters, filter_sizes, num_classes).to(DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    return tokenizer, model, mlb

@st.cache_data
def predict_topk(text: str, _tokenizer, _model, _mlb, top_k: int, max_length: int):
    enc = _tokenizer(
        text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt'
    )
    input_ids = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)
    with torch.no_grad():
        logits = _model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    topk_idx = probs.argsort()[-top_k:][::-1]
    return [( _mlb.classes_[i], float(probs[i]) ) for i in topk_idx]

# Streamlit UI
st.title("Yelp Business Category Predictor")

# Sidebar
st.sidebar.header("Settings")
model_path = st.sidebar.text_input('Model Path', 'yelp_cnn_model.pth')
classes_path = st.sidebar.text_input('Classes JSON Path', 'mlb_classes.json')
max_length = st.sidebar.number_input('Max Seq Length', value=32)
top_k = st.sidebar.number_input('Top K', value=5, min_value=1)

# Load resources
tokenizer, model, mlb = load_resources(model_path, classes_path)

# Input and predict
name = st.text_input('Business Name')
if st.button('Predict') and name:
    with st.spinner('Predicting...'):
        results = predict_topk(name, tokenizer, model, mlb, top_k, max_length)
    st.subheader('Top Predictions')
    for label, prob in results:
        st.write(f"**{label}**: {prob:.4f}")

st.write('_Powered by DistilBERT + CNN_')
