import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from torchvision.datasets import ImageFolder
from torch import nn, optim
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import KFold
import os
import csv

# 檢查 CUDA 是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用設備:", device)

# 創建模型
def create_model(weights_path=None):
    model = models.resnet18()
    num_classes = 2

    # 如果提供了權重路徑，則加載預訓練權重，但不包括全連接層
    if weights_path is not None:
        # 加載預訓練權重
        pretrained_dict = torch.load(weights_path)
        # 獲取模型當前的狀態字典
        model_dict = model.state_dict()
        # 過濾出卷積層的權重
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k != 'fc.weight' and k != 'fc.bias'}
        # 更新當前模型的狀態字典
        model_dict.update(pretrained_dict)
        # 加載過濾後的狀態字典
        model.load_state_dict(model_dict)

    # 修改全連接層以匹配類別數量
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

# 評估模型
def evaluate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches

# 訓練和評估單個群集的函數
def train_and_evaluate(path, model_path, type, k_folds=3):
    # 數據轉換
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加載數據集
    dataset = ImageFolder(root=path, transform=transform)

    # 分割數據集為訓練集和測試集 (9:1)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])



    # 檢查是否已有訓練好的模型
    if os.path.exists(model_path):
        print(f"載入已存在的模型: {model_path}")
        model = create_model(model_path)
    else:
        print(f"開始訓練模型: {type}")
        model = create_model()
        model.to(device)

        best_model = None
        lowest_val_loss = float('inf')

        kfold = KFold(n_splits=k_folds, shuffle=True)

        for fold, (train_ids, _) in enumerate(kfold.split(train_dataset)):
            print(f'{type}, FOLD {fold}')
            print('--------------------------------')

            train_subsampler = SubsetRandomSampler(train_ids)
            train_loader = DataLoader(train_dataset, batch_size=256, sampler=train_subsampler)

            optimizer = optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()

            num_epochs = 50
            last_loss = 0
            trigger_times = 0
            patience = 3
            loss_change_threshold = 0.01

            for epoch in range(num_epochs):
                model.train()
                total_loss = 0.0
                total_batches = 0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    total_batches += 1

                val_loss = evaluate_model(model, DataLoader(test_dataset, batch_size=256, shuffle=True), criterion)
                print(f'Epoch {epoch+1}, Val Loss: {val_loss}')

                loss_change = abs(last_loss - val_loss)
                if loss_change < loss_change_threshold:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print(f'提前停止於 Epoch {epoch+1}')
                        break

                last_loss = val_loss

            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_model = model

        torch.save(best_model.state_dict(), model_path)
        model = best_model

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
    return model, test_dataset, test_loader

def predict_and_save(model, loader, dataset, type, folder='predictions'):
    # 確保模型在正確的設備上
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    if not os.path.exists(folder):
        os.makedirs(folder)

    with torch.no_grad(), open(os.path.join(folder, f'{type}_predictions.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'true_label', 'predicted_label'])

        for images, labels in loader:
            # 確保輸入數據也在正確的設備上
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            for img, true, pred in zip(dataset.dataset.imgs, labels.cpu().numpy(), predicted.cpu().numpy()):
                writer.writerow([os.path.basename(img[0]), true, pred])

    return y_true, y_pred


# 主程序
if __name__ == '__main__':
    best_model_0, test_dataset_0, test_loader_0 = train_and_evaluate('Dataset\clusters\Project_B_clusters_0','clusters_0_resnet_best_model.pth','clusters_0')
    best_model_1, test_dataset_1, test_loader_1 = train_and_evaluate('Dataset\clusters\Project_B_clusters_1','clusters_1_resnet_best_model.pth', 'clusters_1')

    # 確保在 train_and_evaluate 函數中返回 test_dataset 和 test_loader
    y_true_0, y_pred_0 = predict_and_save(best_model_0, test_loader_0, test_dataset_0, 'clusters_0')
    y_true_1, y_pred_1 = predict_and_save(best_model_1, test_loader_1, test_dataset_1, 'clusters_1')

    # 合併群集預測結果
    y_true_combined = y_true_0 + y_true_1
    y_pred_combined = y_pred_0 + y_pred_1

    # 計算合併後準確度
    combined_accuracy = accuracy_score(y_true_combined, y_pred_combined)
    print(f'合併準確度: {combined_accuracy}')
    # 計算 F1 分數
    f1 = f1_score(y_true_combined, y_pred_combined, average='macro')
    print(f'Combined F1 Score: {f1}')
