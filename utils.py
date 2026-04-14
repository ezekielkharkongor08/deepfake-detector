import torch
import torch.nn as nn
import torchaudio
import soundfile as sf

class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def preprocess_audio(file_path):
    import soundfile as sf
    
    # Load audio using soundfile
    waveform, sr = sf.read(file_path)
    waveform = torch.tensor(waveform, dtype=torch.float32)
    
    # Convert to proper shape for torchaudio
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    max_len = target_sr * 3
    if waveform.shape[1] > max_len:
        waveform = waveform[:, :max_len]
    else:
        pad = max_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_mels=64
    )(waveform)

    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel = mel.unsqueeze(0)

    return mel


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu")

    model = AudioClassifier()  

    model.load_state_dict(checkpoint["model_state_dict"], strict=False) 
    model.eval()

    return model
