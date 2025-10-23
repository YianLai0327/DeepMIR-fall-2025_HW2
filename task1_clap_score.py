import laion_clap
import numpy as np
import librosa
import torch
import os
import json
import torch.serialization
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
torch.serialization.add_safe_globals([np.ndarray])

def int16_to_float32(x):
    return (x / 32767.0).astype('float32')


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')

target_dir = "./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s"
ref_dir = "./home/fundwotsai/Deep_MIR_hw2/referecne_music_list_60s"
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

target_files = [file for file in os.listdir(target_dir) if file.endswith(".wav") or file.endswith(".mp3")]
ref_files = []

with open("best_matches.json", "r") as json_file:
    best_matches = json.load(json_file)

for file in target_files:
    ref_file = best_matches[file].get("reference_file", None)
    if ref_file is None:
        print(f"No reference file found for {file}, skipping...")
        continue
    ref_files.append(os.path.join(ref_dir, ref_file))

target_files = [os.path.join(target_dir, file) for file in target_files]
target_audio_embed = model.get_audio_embedding_from_filelist(x = target_files, use_tensor=False)
ref_audio_embed = model.get_audio_embedding_from_filelist(x = ref_files, use_tensor=False)

from sklearn.metrics.pairwise import cosine_similarity

for i, target_file in enumerate(target_files):
    target_embed = target_audio_embed[i].reshape(1, -1)
    ref_embed = ref_audio_embed[i].reshape(1, -1)
    similarity = cosine_similarity(target_embed, ref_embed)[0][0]
    # only store with resolution 4 decimal places
    best_matches[os.path.basename(target_file)]["clap_score"] = round(float(similarity), 4)

with open("suno_clap_score.json", "w") as json_file:
    json.dump(best_matches, json_file, indent=4, ensure_ascii=False)

print("CLAP scores computed and saved to suno_clap_score.json")