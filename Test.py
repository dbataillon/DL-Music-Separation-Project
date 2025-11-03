import os
import numpy as np
import torch
import librosa

from util import LoadAudio, SaveAudio
from configs.config import image_width, patch_size
from U_net import UNetStandard

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    device = get_device()
    print(f"Using device: {device}")

    model_dir = "./model"
    checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not checkpoint_files:
        raise FileNotFoundError("No .pt checkpoint found in ./model")
    latest_ckpt = sorted(checkpoint_files)[-1]
    ckpt_path = os.path.join(model_dir, latest_ckpt)
    print(f"Loading checkpoint: {ckpt_path}")

    # === Load model and weights ===
    model = UNetStandard(num_outputs=4, dropout_p=0.4, final_activation="relu").to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # === Load mixture audio ===
    music_path = "./sample.wav"
    mix_wav_mag, mix_wav_phase = LoadAudio(music_path)

    START = 60
    END = START + patch_size  # 11 seconds (same as TF)
    mix_wav_mag = mix_wav_mag[:, START:END]
    mix_wav_phase = mix_wav_phase[:, START:END]

    # === Prepare input ===
    X = mix_wav_mag[1:].reshape(1, image_width, 128, 1)  # (1, H, 128, 1) same as TF
    X = np.transpose(X, (0, 3, 1, 2))  # NHWC -> NCHW for PyTorch
    X_t = torch.from_numpy(X).float().to(device)

    # === Predict ===
    with torch.no_grad():
        preds = model(X_t, training=False).cpu().numpy()  # (1, 4, H, 128)
    mask = preds[0]  # (4, H, 128)

    # === Save separated stems ===
    stem_names = ["vocal", "bass", "drums", "other"]
    for i, name in enumerate(stem_names):
        target_mag = np.vstack((np.zeros((128,)), mask[i].reshape(image_width, 128)))
        out_path = f"{music_path[:-4]}_{name}.wav"
        SaveAudio(out_path, target_mag, mix_wav_phase)
        print(f"Saved {out_path}")

    print("Done!")

if __name__ == "__main__":
    main()
