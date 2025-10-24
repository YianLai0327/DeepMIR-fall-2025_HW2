import laion_clap
import numpy as np
import librosa
import torch
import os
import json

target_dir = "./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s_index"
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

target_files = [file for file in os.listdir(target_dir) if file.endswith(".wav") or file.endswith(".mp3")]

with open("task2_suno_clap_score.json", "r") as json_file:
    datas = json.load(json_file)
    
# ref_files = [datas[file].get("matched_melodycondition", None) for file in target_files]
ref_files = [datas[file].get("matched_song_in_suno", None) for file in target_files]

target_files = [os.path.join(target_dir, file) for file in target_files]
target_audio_embed = model.get_audio_embedding_from_filelist(x = target_files, use_tensor=False)
ref_audio_embed = model.get_audio_embedding_from_filelist(x = ref_files, use_tensor=False)

print("Embeddings extracted.")

print("Calculating CLAP scores...")
from sklearn.metrics.pairwise import cosine_similarity

for i, target_file in enumerate(target_files):
    target_embed = target_audio_embed[i].reshape(1, -1)
    ref_embed = ref_audio_embed[i].reshape(1, -1)
    similarity = cosine_similarity(target_embed, ref_embed)[0][0]
    # only store with resolution 4 decimal places
    # datas[os.path.basename(target_file)]["melodycondition_clap_score"] = round(float(similarity), 4)
    datas[os.path.basename(target_file)]["suno_clap_score"] = float(f"{similarity:.4f}")

with open("task2_suno_clap_score.json", "w") as json_file:
    json.dump(datas, json_file, indent=4, ensure_ascii=False)

print("CLAP scores computed and saved to suno_clap_score.json")

