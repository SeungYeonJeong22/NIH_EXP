# FROM nvcr.io/nvidia/pytorch:22.09-py3
FROM pytorch/pytorch
RUN pip install scikit-learn tensorboard tqdm einops seaborn randaugment termcolor addict yapf
## RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
ENTRYPOINT [ "python", "./main-causal.py"]
