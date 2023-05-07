Pre-training 

1. Training data
Source domain data: VoxCeleb2 (5994 speakers) with labels
Target domain data: CN-Celeb1 (797speakers) without labels
Noise and reverberation: Musan, RIR

2. Experimental setting
Model: ECAPA-TDNN (https://github.com/TaoRuijie/ECAPA-TDNN)
Loss: AAMsoftmax loss (source domain) + Contrastive loss (target domain)
Data augmentation: add noise and reverb

3. Run the project
Modify the config.yaml.
Run run.py.
