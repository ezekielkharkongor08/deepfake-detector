import torch
import torch.nn as nn
import torchaudio
from torchvision import models
import librosa

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=128
)

def preprocess_audio(file_path):
    
    waveform, sr = librosa.load(file_path, sr=16000)
    waveform = torch.tensor(waveform).unsqueeze(0)

    max_len = 16000 * 6
    
    if waveform.shape[1] > max_len:
        waveform = waveform[:, :max_len]
    else:
        waveform = torch.nn.functional.pad(
            waveform, (0, max_len - waveform.shape[1])
        )

    # mel
    mel = mel_transform(waveform)
    mel = torch.log(mel + 1e-6)
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)

    return mel.unsqueeze(0)

def load_model(path):

    model = models.resnet18(pretrained=False)

    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),

        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),

        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),

        nn.Linear(64, 2)
    )

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"]) 
    model.eval()
    
    return model
