import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from torchvision import models

class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(pretrained=False)

        old_conv = self.model.conv1.weight
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.model.conv1.weight.data = old_conv.mean(dim=1, keepdim=True)

        self.model.fc = nn.Sequential(
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

    def forward(self, x):
        return self.model(x)

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=128,
    n_fft=1024,
    hop_length=512
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

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    max_len = target_sr * 6
    if waveform.shape[1] > max_len:
        waveform = waveform[:, :max_len]
    else:
        pad = max_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    mel = mel_transform(waveform)

    mel = torch.log(mel + 1e-6)
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)

    return mel.unsqueeze(0)

def load_model(path):
    checkpoint = torch.load(path, map_location="cpu")

    model = AudioClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model
