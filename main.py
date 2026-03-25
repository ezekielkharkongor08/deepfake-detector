#!/usr/bin/env python
# coding: utf-8

# Import all the required libraries

# In[4]:


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import torchaudio

from sklearn.model_selection import train_test_split
from torchvision import models


# Loading the audio files

# In[5]:


def load_audio_files(protocol_file, audio_folder):

    audio_paths = []
    labels = []

    with open(protocol_file, "r") as f:

        for line in f:

            parts = line.strip().split()

            file_name = parts[1]
            label = parts[-1]

            audio_path = os.path.join(audio_folder, file_name + ".flac")

            audio_paths.append(audio_path)

            labels.append(0 if label == "bonafide" else 1)

    return audio_paths, labels


# O for real 
# 
# 1 for fake

# Train

# In[6]:


train_audio_folder = "LA/ASVspoof2019_LA_train/flac"
train_protocol_file = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

X_train, y_train = load_audio_files(train_protocol_file, train_audio_folder)


# Validation

# In[7]:


dev_audio_folder = "LA/ASVspoof2019_LA_dev/flac"
dev_protocol_file = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"

X_test, y_test = load_audio_files(dev_protocol_file, dev_audio_folder)


# Test

# In[8]:


eval_audio_folder = "LA/ASVspoof2019_LA_eval/flac"
eval_protocol_file = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"

X_eval, y_eval = load_audio_files(eval_protocol_file, eval_audio_folder)


# Picture

# In[ ]:


sample_path = X_train[0]

waveform, sr = torchaudio.load(sample_path)

plt.figure(figsize=(10,4))
plt.plot(waveform.t().numpy())
plt.title("Waveform of Audio Sample")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


# In[10]:


mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=128
)

mel = mel_transform(waveform)
mel = torch.log(mel + 1e-6)

plt.figure(figsize=(10,4))
plt.imshow(mel[0].numpy(), aspect='auto', origin='lower')
plt.title("Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Mel Frequency")
plt.colorbar()
plt.show()


# Dataset Class

# In[6]:


class ASVSpoofDataset(Dataset):

    def __init__(self, paths, labels):

        self.paths = paths
        self.labels = labels

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=128
        )

        # augmentation for minority class
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        waveform, sr = torchaudio.load(self.paths[idx])

        mel = self.mel_transform(waveform)

        mel = torch.log(mel + 1e-6)

        mel = (mel - mel.mean()) / (mel.std() + 1e-9)

        label = self.labels[idx]

        # AUGMENT ONLY CLASS 0 (bonafide)
        if label == 0:
            mel = self.time_mask(mel)
            mel = self.freq_mask(mel)

        mel = mel.repeat(3,1,1)

        label = torch.tensor(label).long()

        return mel, label


# Using Collate function to have same audio len and dim

# In[7]:


def collate_fn(batch):

    mels = []
    labels = []

    max_len = max(item[0].shape[2] for item in batch)

    for mel, label in batch:

        pad = max_len - mel.shape[2]

        if pad > 0:
            mel = F.pad(mel, (0, pad))

        mels.append(mel)
        labels.append(label)

    mels = torch.stack(mels)
    labels = torch.tensor(labels)

    return mels, labels


# In[8]:


train_dataset = ASVSpoofDataset(X_train, y_train)
test_dataset = ASVSpoofDataset(X_test, y_test)
eval_dataset = ASVSpoofDataset(X_eval, y_eval)


# In[9]:


batch_size = 256


# In[10]:


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)


# Loading the RESNET archi

# In[11]:


model = models.resnet18(pretrained=True)


# Freezing the training for the convolution layers i.e, weights of the conv layers are not updated during training

# In[12]:


for param in model.parameters():
    param.requires_grad = False


# Replacing the RESNET classifier with our own classifier

# In[13]:


model.fc = nn.Sequential(

    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),

    nn.Linear(256, 2)
)


# In[14]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)


# In[ ]:


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


# In[16]:


def train_epoch(loader):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        correct += (preds == y).sum().item()

        total += y.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


# In[17]:


def evaluate(loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == y).sum().item()

            total += y.size(0)

    return correct / total


# In[18]:


epochs = 10

train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(epochs):

    train_loss, train_acc = train_epoch(train_loader)

    test_acc = evaluate(test_loader)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Dev Accuracy: {test_acc:.4f}")

    print("*"*50)


# Plot for accuracy

# In[19]:


plt.figure(figsize=(8,5))

plt.plot(range(1, epochs+1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, epochs+1), test_accuracies, label="Dev Accuracy")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.title("Accuracy vs Epochs")  

plt.legend()
plt.grid(True)

plt.show()


# Plot for Loss

# In[20]:


plt.figure(figsize=(8,5))

plt.plot(range(1, epochs+1), train_losses, label="Train Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.title("Loss vs Epochs")

plt.legend()
plt.grid(True)

plt.show()


# In[21]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

all_preds = []
all_labels = []

model.eval()

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(device)

        outputs = model(x)

        preds = torch.argmax(outputs, dim=1).cpu()

        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.show()


# In[22]:


from sklearn.metrics import classification_report

print(classification_report(all_labels, all_preds))


# In[23]:


eval_accuracy = evaluate(eval_loader)

print("\nFinal Evaluation Accuracy (LA_eval):", eval_accuracy)


# In[24]:


all_preds = []
all_labels = []

model.eval()

with torch.no_grad():

    for x, y in eval_loader:

        x = x.to(device)

        outputs = model(x)

        preds = torch.argmax(outputs, dim=1).cpu()

        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.show()


# In[25]:


print(classification_report(all_labels, all_preds)) 


# In[ ]:





# In[ ]:





# In[ ]:




