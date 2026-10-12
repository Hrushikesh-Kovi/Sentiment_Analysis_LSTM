import re
import warnings
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")


print("Loading Data...")
df = pd.read_csv("Corona Tweet Sentiment Analysis/Corona_NLP_train.csv", encoding="latin1")
test_df = pd.read_csv("Corona Tweet Sentiment Analysis/Corona_NLP_test.csv", encoding="latin1")
df = df[["OriginalTweet", "Sentiment"]]
test_df = test_df[["OriginalTweet", "Sentiment"]]

labels = ["Extremely Negative", "Negative", "Neutral", "Positive", "Extremely Positive"]
df["label"] = df["Sentiment"].map({l: i for i, l in enumerate(labels)})
test_df["label"] = test_df["Sentiment"].map({l: i for i, l in enumerate(labels)})

print("Cleaning Data...")
def clean_text(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"https?\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.strip()

print("Tokenizing...")
lemmatizer = WordNetLemmatizer()

def get_tokens(text):
    tokens = word_tokenize(text)
    lemmas = [
        lemmatizer.lemmatize(token)
        for token in tokens
    ]
    return lemmas

df["cleaned_text"] = df["OriginalTweet"].apply(clean_text)
test_df["cleaned_text"] = test_df["OriginalTweet"].apply(clean_text)
df["tokens"] = df["cleaned_text"].apply(get_tokens)
test_df["tokens"] = test_df["cleaned_text"].apply(get_tokens)

print("Splitting Data...")
train_df, val_df = train_test_split(
    df, random_state=42, test_size=0.2, stratify=df["label"]
)

word_counts = Counter()
for tokens in train_df["tokens"]:
    word_counts.update(tokens)

word_to_id = {
    "<pad>" : 0,
    "<unk>" : 1
}
for word in word_counts:
    word_to_id[word] = len(word_to_id)

embedding_matrix = np.random.normal(0, 0.1, (len(word_to_id), 100)).astype(np.float32)
embedding_matrix[word_to_id["<pad>"]] = 0

print("Loading GloVe Embeddings...")
with open("glove.6B/glove.6B.100d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.rstrip().split()
        word = values[0]

        if word in word_to_id:
            vector = np.asarray(values[1:], dtype=np.float32)
            embedding_matrix[word_to_id[word]] = vector

print("Encoding Tokens...")

def encode_tokens(tokens, pad_dir="right", max_length=50, get_length=False):
    ids = [word_to_id.get(token, word_to_id["<unk>"]) for token in tokens]
    ids = ids[:max_length]

    if len(ids) == 0:
        ids = [word_to_id["<unk>"]]

    length = len(ids)

    if pad_dir.lower() == "right":
        padded_ids = ids + [0] * (max_length - length)
    elif pad_dir.lower() == "left":
        padded_ids = [0] * (max_length - length) + ids
    else:
        raise ValueError("pad_dir must be 'right' or 'left'")

    if get_length:
        return padded_ids, length

    return padded_ids

print("Making Loaders...")
def make_loader_with_lengths(data, shuffle, padding_direction="left"):
    encoded = [
        encode_tokens(tokens, pad_dir=padding_direction, get_length=True) 
        for tokens in data["tokens"]
    ]
    X = torch.LongTensor([item[0] for item in encoded])
    y = torch.LongTensor(data["label"].values)
    lengths = torch.LongTensor([item[1] for item in encoded])
    dataset = TensorDataset(
        X, lengths, y
    )
    return DataLoader(dataset, shuffle=shuffle, batch_size=64)

def make_loader_without_lengths(data, shuffle, padding_direction="left"):
    encoded = [
        encode_tokens(tokens, pad_dir=padding_direction)
        for tokens in data["tokens"]
    ]
    X = torch.LongTensor([item for item in encoded])
    y = torch.LongTensor(data["label"].values)
    dataset = TensorDataset(
        X, y
    )
    return DataLoader(dataset, shuffle=shuffle, batch_size=64)

train_loader = make_loader_without_lengths(train_df, True)
val_loader = make_loader_without_lengths(val_df, False)
test_loader = make_loader_without_lengths(test_df, False)

print("Building Model...")
class LSTM(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix),
            freeze=False,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=100,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embeddings(x)
        _, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        output = self.fc(last_hidden)
        return output


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
model = LSTM(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode="min", factor=0.5, patience=2
)

print("Training Model...")
epochs = 20
patience = 5
best_model_state = None
epochs_without_improvement = 0
best_val_loss = float("inf")

training_loss = []
validation_loss = []
epoch_count = 0

for epoch in range(epochs):
    model.train()
    train_total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_total_loss += loss.item()
    train_loss = train_total_loss / len(train_loader)
    training_loss.append(train_loss)

    model.eval()
    val_total_loss = 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            val_total_loss += loss.item()

    val_loss = val_total_loss / len(val_loader)
    validation_loss.append(val_loss)
    epoch_count += 1

    scheduler.step(val_loss)

    print(f"Epoch : {epoch+1}/{epochs}  |  Training Loss : {train_loss:.4f}  |  Validation Loss : {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = {
            key : value.detach().clone()
            for key, value in model.state_dict().items()
        }
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after epoch {epoch+1}")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

print("Evaluating Model...")
model.eval()
predictions = []
actual_vals = []

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        actual_vals.extend(y.cpu())
        logits = model(X)
        preds = torch.argmax(logits, dim=1).cpu()
        predictions.extend(preds)

print("Classification Report...")
print(classification_report(actual_vals, predictions))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(actual_vals, predictions), annot=True, fmt="d", cmap="mako")
plt.title("Confusion Matrix", fontweight=600, fontsize=24)
plt.tight_layout()

plt.figure(figsize=(6, 5))
sns.lineplot(x=range(1, epoch_count+1), y=training_loss, label="Training Loss")
sns.lineplot(x=range(1, epoch_count+1), y=validation_loss, label="Validation Loss")
plt.xlabel("Epoch", fontweight=600)
plt.ylabel("Loss", fontweight=600)
plt.axhline(min(validation_loss), ls="--", c="green")
plt.title("Training vs Validation Loss", fontweight=600, fontsize=18)
plt.xticks(list(range(1, epoch_count+1)))
plt.legend()
plt.tight_layout()
plt.show()

torch.save(model.state_dict(), "simple_lstm_corona_sentiment_analysis.pth")

print("Program Execution Completed...")