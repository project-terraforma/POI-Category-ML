import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score


# === 1. Define a CNN model ===
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # Assume input images are 3×64×64 → after 3 pools → 128×8×8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x

# === 2. Training function ===
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

# === 3. Validation function ===
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

# === 4. Test function (same as validation, but kept separate if you want to save results) ===
def test(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_acc = accuracy_score(all_labels, all_preds)
    return test_acc

# === 5. Putting it all together in a main training loop ===
def main():
    # Hyperparameters
    num_epochs = 20
    batch_size = 64
    learning_rate = 1e-3
    num_classes = 10  # adjust to your number of classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 5.1. Create Datasets and DataLoaders ---
    # Replace `YourTrainDataset`, `YourValDataset`, `YourTestDataset` with actual dataset classes.
    # For example, if you already have an OvertureDataset, you can do:
    # train_dataset = OvertureDataset(train_data, tokenizer, max_length, mlb)
    # val_dataset   = OvertureDataset(val_data, tokenizer, max_length, mlb)
    # test_dataset  = OvertureDataset(test_data, tokenizer, max_length, mlb)
    #
    # Here, we assume images-based dataset for CNN, so use something like torchvision.datasets.ImageFolder or a custom Dataset


    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root="path/to/train", transform=transform)
    val_dataset   = datasets.ImageFolder(root="path/to/val", transform=transform)
    test_dataset  = datasets.ImageFolder(root="path/to/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- 5.2. Instantiate model, loss, optimizer ---
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    best_model_wts = None

    # --- 5.3. Epoch loop ---
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()
            torch.save(best_model_wts, "best_cnn_model.pth")

    # --- 5.4. Load best model for testing ---
    model.load_state_dict(torch.load("best_cnn_model.pth"))
    test_acc = test(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")

def load_data():
    bbox = 37.29, -121.83, 37.35, -121.98
    


if __name__ == "__main__":
    main()
