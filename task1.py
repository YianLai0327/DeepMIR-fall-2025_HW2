import torch
import soundfile as sf
from stable_audio_tools import get_pretrained_model
import os
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_embedding(audio_path, autoencoder):
    audio, sr = sf.read(audio_path)
    if audio.shape[0] != sr * 60:
        print(f"Audio length is {audio.shape[0]/sr} seconds, expected 60 seconds.")
        return None

    # print(f"Original shape: {audio.shape}")  # (samples, channels)
    # print(f"Sampling rate: {sr}")

    audio_tensor = torch.from_numpy(audio).float()
    audio_tensor = audio_tensor.T

    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    audio_tensor = audio_tensor.unsqueeze(0)

    # print(f"Processed audio: {audio_tensor.shape}")

    target_sr = cfg["sample_rate"]
    if sr != target_sr:
        print(f"Resample: {sr} Hz -> {target_sr} Hz")
        import torchaudio.functional as F
        audio_tensor = F.resample(audio_tensor, sr, target_sr)
        print(f"Resampled audio shape: {audio_tensor.shape}")

    # move to device
    audio_tensor = audio_tensor.to(device)


    # print("extract embedding...")
    with torch.no_grad():
        latent = autoencoder.encode(audio_tensor)

    # print(f"Latent shape: {latent.shape}")

    return latent

target_dir = "./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s"
ref_dir = "./home/fundwotsai/Deep_MIR_hw2/referecne_music_list_60s"

if os.path.exists("target_latents.pt") and os.path.exists("ref_latents.pt"):
    print("already extracted latents, skip loading model")
else:
    print("loading...")
    model, cfg = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to(device)
    autoencoder = model.pretransform.model

    target_latents = []
    ref_latents = []

target_files = [file for file in os.listdir(target_dir) if file.endswith(".wav") or file.endswith(".mp3")]
ref_files = [file for file in os.listdir(ref_dir) if file.endswith(".wav") or file.endswith(".mp3")]

# only extract when not extracted before
if os.path.exists("target_latents.pt"):
    print("loading existing target latents...")
    target_latents = torch.load("target_latents.pt")
else:
    print("no latents yet, extracting target latents...")
    for (i, file) in enumerate(os.listdir(target_dir)):
        print(f"processing {i+1}/{len(os.listdir(target_dir))}: {file}")
        if file.endswith(".wav") or file.endswith(".mp3"):
            # audio_path = f"{target_dir}/6_rock_102_beat_3-4.wav"
            audio_path = os.path.join(target_dir, file)
            latent = extract_embedding(audio_path, autoencoder)
            if latent is not None:
                target_latents.append(latent)
            else:
                print(f"Skipping {file} since the song is not 60s long")
    # save the target latents
    torch.save(target_latents, "target_latents.pt")

if os.path.exists("ref_latents.pt"):
    print("loading existing reference latents...")
    ref_latents = torch.load("ref_latents.pt")
else:
    print("no latents yet, extracting reference latents...")
    for (i, file) in enumerate(os.listdir(ref_dir)):
        print(f"processing {i+1}/{len(os.listdir(ref_dir))}: {file}")
        if file.endswith(".wav") or file.endswith(".mp3"):
            # audio_path = f"{ref_dir}/6_rock_102_beat_3-4.wav"
            audio_path = os.path.join(ref_dir, file)
            latent = extract_embedding(audio_path, autoencoder)
            if latent is not None:
                ref_latents.append(latent)
            else:
                print(f"Skipping {file} since the song is not 60s long")
    # save the reference latents
    torch.save(ref_latents, "ref_latents.pt")

print("all latents extracted.")

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def extract_global_features(latent):
    """
    從 latent 提取全域特徵
    輸入: [batch, channels, time, features]
    輸出: [batch, channels * features] (對時間維度平均)
    """
    # 對時間維度取平均
    latent_pooled = latent.mean(dim=2)  # [batch, channels, features]
    # Flatten 通道和特徵維度
    features = latent_pooled.reshape(latent_pooled.shape[0], -1)
    return features

# 處理 target latents
target_latents_cat = torch.cat(target_latents, dim=0)
target_features = extract_global_features(target_latents_cat).cpu().numpy()

# 處理 reference latents
ref_latents_cat = torch.cat(ref_latents, dim=0)
ref_features = extract_global_features(ref_latents_cat).cpu().numpy()

print(f"Target features shape: {target_features.shape}")
print(f"Reference features shape: {ref_features.shape}")

# 計算相似度
similarity_matrix = cosine_similarity(target_features, ref_features)

# find top-3 matches for each target
best_matches = np.argmax(similarity_matrix, axis=1)
best_scores = np.max(similarity_matrix, axis=1)

top_k = 5
top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -top_k:][:, ::-1]
top_k_scores = np.sort(similarity_matrix, axis=1)[:, -top_k:][:, ::-1]

print("\n每個 Target 的 Top-5 匹配:")
print("=" * 80)
for i in range(len(target_files)):
    print(f"\nTarget {i}: {target_files[i]}")
    for rank, (match_idx, score) in enumerate(zip(top_k_indices[i], top_k_scores[i]), 1):
        print(f"  {rank}. {ref_files[match_idx]:50s} (相似度: {score:.4f})")

# 顯示結果
print("\n最佳匹配 (Top-1):")
print("=" * 80)
for i, (match_idx, score) in enumerate(zip(best_matches, best_scores)):
    target_name = target_files[i].split('/')[-1] if '/' in target_files[i] else target_files[i]
    ref_name = ref_files[match_idx].split('/')[-1] if '/' in ref_files[match_idx] else ref_files[match_idx]
    print(f"{i+1:3d}. {target_name:40s} → {ref_name:40s} (相似度: {score:.4f})")

# 儲存結果
with open("best_matches.txt", "w") as f:
    f.write("target_file,reference_file,similarity_score\n")
    for i, (match_idx, score) in enumerate(zip(best_matches, best_scores)):
        f.write(f"{target_files[i]},{ref_files[match_idx]},{score:.4f}\n")

def plot_topk_matches(similarity_matrix, target_files, ref_files, k=5, save_path="topk_matches.png"):
    """繪製每個 target 的 top-k 匹配"""
    
    n_targets = len(target_files)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten() if n_targets > 1 else [axes]
    
    for i, ax in enumerate(axes):
        if i < n_targets:
            # 取得 top-k 匹配
            top_k_indices = np.argsort(similarity_matrix[i])[-k:][::-1]
            top_k_scores = similarity_matrix[i][top_k_indices]
            
            # 繪製長條圖
            colors = plt.cm.RdYlGn(top_k_scores)  # 根據分數上色
            bars = ax.barh(range(k), top_k_scores, color=colors)
            ax.set_yticks(range(k))
            ax.set_xlabel('Similarity Score', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_title(f'Target: {i+1}.wav', 
                        fontsize=10, fontweight='bold')
            ax.invert_yaxis()
            
            # 在長條上標註分數
            for j, (bar, score) in enumerate(zip(bars, top_k_scores)):
                ax.text(score + 0.02, j, f'{score:.3f}', 
                       va='center', fontsize=8)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top-K 匹配圖已儲存: {save_path}")

# 使用
plot_topk_matches(similarity_matrix, target_files, ref_files, k=5)




