import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from torchvision import models

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=128
)

def preprocess_audio(file_path):

    try:
        waveform, sr = torchaudio.load(file_path)
    except:
        waveform, sr = sf.read(file_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.mean(dim=1, keepdim=True).T

    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample
    target_sr = 16000
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

    # pad / truncate (6 sec)
    max_len = target_sr * 6
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

    return mel.unsqueeze(0)  # (1, 1, H, W)

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
