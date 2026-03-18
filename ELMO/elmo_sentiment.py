# ELMo情感分析
# 安装依赖: pip install allennlp torch pandas scikit-learn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from allennlp.modules.elmo import Elmo, batch_to_ids

# ELMo预训练模型
OPTIONS_FILE = "elmo_options.json"
WEIGHT_FILE  = "elmo_weights.hdf5"
ELMO_DIM = 256  # ELMo输出维度 = 128 * 2 (双向)


# ── 数据集 ──────────────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = [t.split() for t in texts]   # 简单空格分词
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch):
    texts, labels = zip(*batch)
    char_ids = batch_to_ids(list(texts))           # ELMo字符级ID
    return char_ids, torch.tensor(labels, dtype=torch.long)


# ── 模型 ────────────────────────────────────────────────────────────────────

class ELMoClassifier(nn.Module):
    """ELMo嵌入 + 平均池化 + MLP分类器"""

    def __init__(self, elmo_dim=ELMO_DIM, hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.elmo = Elmo(OPTIONS_FILE, WEIGHT_FILE,
                         num_output_representations=1, dropout=0)
        self.classifier = nn.Sequential(
            nn.Linear(elmo_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, char_ids):
        out = self.elmo(char_ids)
        reps = out['elmo_representations'][0]      # (B, T, D)
        mask = out['mask'].float()                 # (B, T)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (reps * mask.unsqueeze(-1)).sum(dim=1) / lengths
        return self.classifier(pooled)


# ── 训练 / 评估 ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for char_ids, labels in loader:
        char_ids, labels = char_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(char_ids), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for char_ids, labels in loader:
            logits = model(char_ids.to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            trues.extend(labels.tolist())
    return preds, trues


def predict(model, texts, device):
    model.eval()
    char_ids = batch_to_ids([t.split() for t in texts]).to(device)
    with torch.no_grad():
        return torch.argmax(model(char_ids), dim=1).cpu().tolist()


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载CSV数据
    df = pd.read_csv('sentiment_dataset_1000.csv')
    texts, labels = df['text'].tolist(), df['label'].tolist()
    print(f"数据总量: {len(texts)} 条  |  标签分布: {pd.Series(labels).value_counts().to_dict()}")

    # 划分训练/验证集
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tr_loader = DataLoader(SentimentDataset(tr_texts, tr_labels),
                           batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(SentimentDataset(val_texts, val_labels),
                            batch_size=16, collate_fn=collate_fn)

    # 初始化模型
    model = ELMoClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练
    epochs = 5
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, tr_loader, optimizer, criterion, device)
        preds, trues = evaluate(model, val_loader, device)
        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average='weighted')
        print(f"Epoch {epoch}/{epochs}  Loss: {loss:.4f}  Acc: {acc:.4f}  F1: {f1:.4f}")

    # 详细报告
    print("\nClassification Report:")
    print(classification_report(trues, preds, target_names=['Positive', 'Negative']))

    # 示例预测
    label_map = {0: 'Positive', 1: 'Negative'}
    samples = [
        "I love this product!",
        "This is terrible, I hate it.",
        "Feeling great today!",
        "I'm so disappointed with the results.",
    ]
    print("--- 示例预测 ---")
    for text, pred in zip(samples, predict(model, samples, device)):
        print(f"  [{label_map[pred]}]  {text}")

    # 保存模型
    torch.save(model.state_dict(), 'elmo_sentiment_model.pth')
    print("\n模型已保存: elmo_sentiment_model.pth")


if __name__ == '__main__':
    main()
