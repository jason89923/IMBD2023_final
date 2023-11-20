import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torch import nn, optim
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torchvision.models import ResNet18_Weights

# 指示變量，決定是否加載已訓練模型
load_pretrained_model = False
model_path = 'resnet_model.pth'  # 已訓練模型的路徑


# 確認 CUDA 是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def create_model():
    # 使用更新的方式加载预训练模型
    weights = ResNet18_Weights.IMAGENET1K_V1  # 选择预训练权重
    model = models.resnet18(weights=weights)
    num_classes = 2  # 假设有两类
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def evaluate_model(model, val_loader, criterion):
    model.eval()  # 将模型设置为评估模式
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():  # 在这个块中不计算梯度
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_batches += 1

    average_loss = total_loss / total_batches
    return average_loss

# 根據條件選擇加載模型還是創建新模型
if load_pretrained_model:
    # 加載已訓練的模型
    try:
        resnet = create_model()
        resnet.load_state_dict(torch.load(model_path))
        resnet = resnet.to(device)
        print("Loaded pretrained model.")
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Creating a new model.")
        resnet = create_model()
        resnet = resnet.to(device)
else:
    # 創建新模型
    resnet = create_model()
    resnet = resnet.to(device)
    print("Created a new model.")

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters())

# 數據轉換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加載數據集
dataset = ImageFolder(root='Dataset\clusters\Project_B_clusters_0', transform=transform)  # 替換為您的數據集路徑
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 訓練模型
num_epochs = 50
for epoch in range(num_epochs):
    resnet.train()
    total_loss = 0.0  # 初始化總損失為 0
    total_batches = 0  # 初始化批次計數
    best_loss = float('inf')
    patience = 2
    trigger_times = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # 累計每個批次的損失
        total_batches += 1  # 累計批次數

    average_loss = total_loss / total_batches  # 計算平均損失
    val_loss = evaluate_model(resnet, test_loader, criterion)
    print(f'Epoch {epoch+1}, Val Loss: {val_loss}')

    if val_loss < best_loss:
        best_loss = val_loss
        
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# 評估模型
resnet.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet(inputs)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# 儲存模型參數
torch.save(resnet.state_dict(), 'resnet_model.pth')

print("Model saved successfully!")

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
