# deepfake-detector

This project tries to classify audio as either real (bonafide) or fake (spoofed).  
It uses a deep learning model trained on spectrograms of audio signals.

I used ResNet18 with transfer learning.  
Instead of images, the model takes Mel Spectrograms generated from audio.

ASVspoof2019 (Logical Access)

- Load audio files using torchaudio
- Convert them into Mel Spectrograms
- Normalize the data
- Train a ResNet model to classify real vs fake audio


## Results
The model is able to distinguish between real and fake audio with decent accuracy.  
-Train Loss: 0.0450
-Train Accuracy: 0.9854
-Dev Accuracy: 0.9844

-Final Evaluation Accuracy (LA_eval): 0.9610455240956245

## Notes
- Dataset is not included because it is too large
- Models are also not uploaded
