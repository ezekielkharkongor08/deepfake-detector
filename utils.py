import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from torchvision import models

class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Load ResNet18
        self.model = models.resnet18(pretrained=False)
        
        # Modify first conv layer for single channel input
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
        
        # Replace classifier
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


def preprocess_audio(file_path):
    import soundfile as sf
    
    waveform, sr = sf.read(file_path)
    waveform = torch.tensor(waveform, dtype=torch.float32)
  
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # 6 seconds for ResNet18
    max_len = target_sr * 6
    if waveform.shape[1] > max_len:
        waveform = waveform[:, :max_len]
    else:
        pad = max_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    # 128 mel bands for ResNet18
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_mels=128
    )(waveform)

    mel = torch.log(mel + 1e-6)
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)
    mel = mel.unsqueeze(0)

    return mel


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu")

    model = AudioClassifier()  

    model.load_state_dict(checkpoint["model_state_dict"], strict=False) 
    model.eval()

    return model
